import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import numpy as np

from tqdm import tqdm
import time
import torch.nn.functional as F
from A_MM_KG_Agent._2_entity_res_anatomy_disorder import preprocess_report, merge_entities
from A_MM_KG_Agent._3_three import load_graph, extract_entity_links
from A_MM_KG_Agent._4_three_bio_clinicalbert import build_features_for_entity
from A_MM_KG_Agent._6_GraphAttentionEmbedder import GraphAttentionEmbedder
from A_RAG_Agent.rag_entity_merge import EntityWiseBGESearcher, merge_dedup_chunks_only
from A_RAG_Agent.z_chunks_clinicalbert import encode_texts
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT/"
from transformers import AutoTokenizer, AutoModel 

CHUNKS_JSON_PATH = "A_RAG_Agent/chunks.json"
BGE_M3_PATH = "A_RAG_Agent/bge-m3"
BGE_RERANKER_PATH = "A_RAG_Agent/bge-reranker-v2-m3"
# Doc_vecs_cache_pt = "/wangx/home/E24301191/mycode/R2GenGPT-main/R2GenGPT-main/A_RAG_Agent/doc_vecs_cache.pt"
DOC_VECS_CACHE_PT = "A_RAG_Agent/doc_vecs_cache.pt"


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
       
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        
        print(f'Loading Bio_ClinicalBERT:{MODEL_NAME}')
        self.Bio_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.Bio_ClinicalBERT = AutoModel.from_pretrained(MODEL_NAME)
        self.Bio_ClinicalBERT.eval()

        # 显式冻结参数，避免 DDP 报未使用参数
        for p in self.Bio_ClinicalBERT.parameters():
            p.requires_grad = False

        print(f'Loading vision encoder:{args.vision_model}')

        # ======== Bio_ClinicalBERT ========  


        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        print('Loading EntityWiseBGESearcher')
        # ======== EntityWiseBGESearcher 对齐层 ========
        self.searcher = EntityWiseBGESearcher(
            bge_m3_path=BGE_M3_PATH,
            bge_reranker_path=BGE_RERANKER_PATH,
            doc_vecs_cache_path=DOC_VECS_CACHE_PT,
            device=torch.cuda.current_device(),   # 例如 "cuda:0"
            max_length=512,
            normalize_cache=True,
        )

        print('Loading Conceptformer')
        self.gat_embedder = GraphAttentionEmbedder(
            node_dim=768,     # 节点 embedding 维度
            edge_dim=768,     # 边 embedding 维度
            hidden_dim=256,              # 隐藏层大小
            output_dim=768,              # 最终输出维度
            num_layers=2,                # MLP 层数
            num_neighbors=5,            # 每个节点邻居数
            activation="gelu",           # 激活函数
            num_pseudo_words=1           # 注意力头数
            )
        self.concept_proj = nn.Linear(768, self.visual_encoder.num_features)
        # self.concept_ln   = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.prompt_concept = "Graph knowledge: <KG><KGHere></KG>"
        res_dict_file = 'A_MM_KG_Agent/res_dict_aliases.json'    # 图谱实体，用来提取实体
        with open(res_dict_file, 'r') as file:
            self.res_dict_id = json.load(file)
        relations_file = "A_MM_KG_Agent/kg_relations.csv"     # 图谱关系，用来提取三元组（子图）
        self.relations = load_graph(relations_file)
        # ======== 新增：Conceptformer 对齐层 =======

        # ======== 加载一阶段权重 ========
        self.load_from_pretrained(url_or_filename='/wangx_nas/QYH/Retrival/CheXpert_plus/baseline/v1_deep/checkpoints/checkpoint_epoch6_step16432_bleu0.101938_cider0.208091.pth')  

        if args.delta_file is not None:
            # state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'), weights_only=True)
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def load_from_pretrained(self, url_or_filename):
        # 判断传入的参数 url_or_filename 是否是一个文件路径
        if os.path.isfile(url_or_filename):
            # 如果是文件路径，加载文件到 CPU
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            # 如果路径无效，则抛出一个 RuntimeError 异常
            raise RuntimeError("checkpoint url or path is invalid")

        # 从加载的 checkpoint 中提取出 "model" 键对应的部分。
        # 这通常是一个包含模型参数（权重）的字典，state_dict 保存了模型的所有参数。
        state_dict = checkpoint["model"]

        # 调用当前模型的 load_state_dict 方法，将从 checkpoint 中加载的权重（state_dict）加载到当前模型中。
        # strict=False 表示如果加载的权重字典和模型的结构不完全匹配（比如某些键缺失或者名字不匹配），不会抛出错误，只是那些未匹配的权重不会加载。
        # 这种情况通常发生在加载预训练模型时，保存的模型架构和当前模型架构略有不同，但大部分权重依然能够成功加载。
        msg = self.load_state_dict(state_dict, strict=False)
        # 打印加载成功的信息，表示从指定的路径加载了 checkpoint
        print("load checkpoint from %s" % url_or_filename)
        return msg

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    
    # ======== 新增：编码 Conceptformer 输出 ========
    def encode_concept(self, Draft_text, res_dict_id, relations):
        device = self.device
        Bio_tokenizer = self.Bio_tokenizer
        Bio_ClinicalBERT = self.Bio_ClinicalBERT
        outputs_all = torch.empty(0, 100, 768).to(device)  # 初始化为空张量

        for Draft in Draft_text:
            # 确保所有输入数据在正确设备上
            Draft = Draft.to(device) if isinstance(Draft, torch.Tensor) else Draft
        
            result = preprocess_report(Draft, res_dict_id)
            merged_entities = merge_entities(result)
            # 查看merged_entities中的内容
            # print(merged_entities)
            entity_links = extract_entity_links(relations, merged_entities, topk=10)  # 消融实验修改 topk
    
            out_append = []
            for central in entity_links.keys():
                neighbors = entity_links.get(central, [])
            
                # 确保实体数据在正确设备上
                central = central.to(device) if isinstance(central, torch.Tensor) else central
                neighbors = [n.to(device) if isinstance(n, torch.Tensor) else n for n in neighbors]
            
                central_emb, neighbor_embs, edge_embs = build_features_for_entity(central, neighbors, device, Bio_ClinicalBERT, Bio_tokenizer)
            
                # 确保返回的嵌入在正确设备上
                central_emb = central_emb.to(device)
                neighbor_embs = neighbor_embs.to(device)
                edge_embs = edge_embs.to(device)
            
                # 补batch维
                central_b = central_emb.unsqueeze(0)   # (1, 1, D)
                neighbors_b = neighbor_embs.unsqueeze(0)  # (1, topk, D)
                edges_b = edge_embs.unsqueeze(0)
                out = self.gat_embedder(central_b, neighbors_b, edges_b)
                out_append.append(out)

            outputs = torch.cat(out_append, dim=1)

            # 动态调整填充长度
            max_len = max(outputs.shape[1], outputs_all.shape[1])
            outputs_padded = F.pad(outputs, (0, 0, 0, max_len - outputs.shape[1]))
            outputs_all_padded = F.pad(outputs_all, (0, 0, 0, max_len - outputs_all.shape[1]))
        
            outputs_all = torch.cat((outputs_padded, outputs_all_padded), dim=0) 
        return outputs_all

    # ======== 新增：编码 Conceptformer 输出 ========
    def encode_rag(self, Draft_text, res_dict_id, relations, pad_to_multiple_of: int = 1, max_len: int = None):
        """
        返回:
        outputs_all: (B, Nmax, 768)  # B=batch size, Nmax=padding后的最大长度
        attn_mask : (B, Nmax)        # 1=有效token/chunk, 0=padding
        """
        device = self.device
        Bio_tokenizer = self.Bio_tokenizer
        Bio_ClinicalBERT = self.Bio_ClinicalBERT

        all_emb_list = []   # 每个元素: (n_i, 768)
        all_len = []

        for Draft in Draft_text:
            if isinstance(Draft, torch.Tensor):
                Draft = Draft.to(device)

            result = preprocess_report(Draft, res_dict_id)
            merged_entities = merge_entities(result)

            # 1) 从外部知识库中进行向量检索
            retrieved = self.searcher.run(
                report_text=Draft,
                merged_entities=merged_entities,
                chunks_json_path=CHUNKS_JSON_PATH,
                top_n=10,
                top_k=1,
                rerank_batch_size=8,
            )

            # retrieved 是 per-entity 的 dict，需要先变成 chunk 文本列表再送进 encode_texts
            merged_chunks = merge_dedup_chunks_only(retrieved)   # {"chunks":[...]}
            chunk_texts = merged_chunks["chunks"]                # List[str]

            # 2) 编码 chunk 文本 => embeddings
            embeddings = encode_texts(
                chunk_texts,
                Bio_tokenizer,
                Bio_ClinicalBERT,
                device,
                batch_size=16
            )

            # -------- 关键：统一成 (n,768) --------
            if embeddings is None:
                emb2d = torch.empty(0, 768, device=device)
            else:
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings)

                if not torch.is_tensor(embeddings):
                    raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

                embeddings = embeddings.to(device)

                if embeddings.dim() == 3:
                    emb2d = embeddings.squeeze(0)
                elif embeddings.dim() == 2:
                    emb2d = embeddings
                else:
                    raise ValueError(f"Unexpected embeddings shape: {tuple(embeddings.shape)}")


            all_emb_list.append(emb2d)
            all_len.append(emb2d.size(0))

        # -------- 动态 padding：得到 Nmax --------
        B = len(all_emb_list)
        if B == 0:
            return torch.empty(0, 0, 768, device=device), torch.empty(0, 0, device=device, dtype=torch.long)

        Nmax = max(all_len) if max_len is None else int(max_len)

        # 可选：pad 到某个倍数（比如 transformer 里喜欢 pad_to_multiple_of=8/16）
        if pad_to_multiple_of > 1:
            Nmax = int((Nmax + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of)

        # -------- pad + stack --------
        padded_list = []
        attn_masks = []

        for emb2d, n in zip(all_emb_list, all_len):
            if n > Nmax:
                # 如果强行设了 max_len 更小：截断
                emb2d = emb2d[:Nmax, :]
                n = Nmax

            pad_len = Nmax - n
            if pad_len > 0:
                # (n,768) -> (Nmax,768)
                emb2d = F.pad(emb2d, pad=(0, 0, 0, pad_len), value=0.0)

            # (1,Nmax,768)
            padded_list.append(emb2d.unsqueeze(0))

            # attention mask: (Nmax,)
            mask = torch.zeros(Nmax, device=device, dtype=torch.long)
            mask[:n] = 1
            attn_masks.append(mask.unsqueeze(0))

        outputs_all = torch.cat(padded_list, dim=0)  # (B, Nmax, 768)
        attn_mask = torch.cat(attn_masks, dim=0)     # (B, Nmax)

        return outputs_all


    # ======== 新增：编码 Conceptformer 输出 ========

   # 将图像编码为特征向量，以供后续的文本生成模型使用
    def encode_img(self, images, Draft_text, res_dict_id, relations):
        image_embeds = []          # 图像特征向量

        concept_embeds = self.encode_concept(Draft_text, res_dict_id, relations)
        rag_embeds = self.encode_rag(Draft_text, res_dict_id, relations)
        for image in images:  # 遍历 images 列表中的每一张图片。images 是一个包含多个图像的批量数据（batch）
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device) # 给嵌入添加一个新的维度，在第二维（维度 1）增加一个维度。
                
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)  
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        # ======== 新增：编码 Conceptformer 特征 ========
        concept_embeds = self.concept_proj(concept_embeds)
        rag_embeds = self.concept_proj(rag_embeds)

        image_embeds = torch.cat([image_embeds, rag_embeds, concept_embeds], dim=1) # 将视觉特征和图谱特征进行拼接

        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        # ======== 新增：编码 Conceptformer 特征 ========
        return inputs_llama, atts_llama

  # 将图像特征嵌入（embeddings）与预定义的提示文本组合起来。
    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    def forward(self, samples):
        image = samples["image"]

        # ======== 新增：编码 Conceptformer 特征 ========
        Draft_text = samples["Draft_text"]  # 初步报告
        # print("Draft_text_forward:", Draft_text)
        res_dict_id =self.res_dict_id
        relations =self.relations
        img_embeds, atts_img = self.encode_img(image, Draft_text, res_dict_id, relations)
        # ======== 新增：编码 Conceptformer 特征 ========

        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)  # 视觉特征和文本特征会被合并在一起，形成最终的输入嵌入
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)  # 注意力掩码

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]

        # ======== 新增：编码 Conceptformer 特征 ========
        Draft_text = samples["Draft_text"]  # 初步报告
        # print("Draft_text_validation_step:", Draft_text)
        res_dict_id =self.res_dict_id
        relations =self.relations
        img_embeds, atts_img = self.encode_img(image, Draft_text, res_dict_id, relations)
        # ======== 新增：编码 Conceptformer 特征 ========

        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        # if self.trainer.local_rank == 0:
        #     if val_score > self.val_score:
        #         self.save_checkpoint(eval_res)
        #         self.val_score = val_score
        # self.val_step_outputs.clear()

        if self.trainer.local_rank == 0:
                # **只保存Bleu_4大于100的模型**
            if eval_res.get("Bleu_4", 0) > 0.095:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]

        # ======== 新增：编码 Conceptformer 特征 ========
        Draft_text = samples["Draft_text"]  # 初步报告
        res_dict_id =self.res_dict_id
        relations =self.relations
        img_embeds, atts_img = self.encode_img(image, Draft_text, res_dict_id, relations)
        # ======== 新增：编码 Conceptformer 特征 ========

        # img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()