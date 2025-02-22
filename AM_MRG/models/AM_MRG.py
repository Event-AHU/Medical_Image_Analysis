import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from models.Qformer import BertConfig, BertLMHeadModel
from hopfield_layers.hflayers import HopfieldLayer
import pickle
import numpy as np
import random
import torch.nn.functional as F
from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16

class AMMRG(pl.LightningModule):
    """
    AMMRG model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.type = args.type
        self.stageone_path = args.stageone_path
        maps_num = args.maps_num
        report_total_samples = args.report_total_samples
        report_memory_path = args.report_memory_path
        REPORT_LABELS = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = arm_large_pz16(self.type)
        finetune = args.vision_model
        if finetune!='None':
            checkpoint = torch.load(finetune, map_location='cpu')
            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            checkpoint_model = checkpoint['model']
            new_dict = {}
            for k, v in checkpoint_model.items():
                if "visual_encoder." in k:
                    new_dict[k.replace("visual_encoder.", "")] = v
            # load pre-trained model
            self.visual_encoder.load_state_dict(new_dict)
         print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        self.Qformer_proj = nn.Linear(self.visual_encoder.num_features, 1408)
        print('Loading Q-Former')
        self.Qformer, self.disease_tokens = self.init_Qformer(14, 1408, False)
        self.load_from_pretrained(url_or_filename='configs/blip2_pretrained_flant5xxl.pth')  # load q-former weights here

        img_f_dim = self.Qformer.config.hidden_size # 768
        self.output_proj = nn.Linear(img_f_dim, self.visual_encoder.num_features)
        self.output_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.output_proj.bias.data.zero_()
        
        diseaseaware_token = self.load_disease_token(self.stageone_path)
        with open(args.class_activation_maps_path, 'rb') as f:
	        class_activation_maps = pickle.load(f)
        all_arrays = [array for value_list in class_activation_maps.values()
              for array in (random.sample(list(value_list), maps_num) if len(value_list) > maps_num else value_list)]
        cam_features = torch.from_numpy(np.stack(all_arrays)).unsqueeze(0).to(diseaseaware_token.device)  # [1, N, 768]
        self.diseaseaware_token = torch.cat((diseaseaware_token, cam_features), dim=1)  # [1, N + 14, 768]

        with open(report_memory_path, 'rb') as f:
	        report_memory = pickle.load(f)
        sample_counts = {label: len(report_memory[label]) for label in REPORT_LABELS}
        total_sample_count = sum(sample_counts.values())
        
        weights = {label: count / total_sample_count for label, count in sample_counts.items()}
        samples_per_label = {label: int(weight * report_total_samples) for label, weight in weights.items()}
        adjustment = report_total_samples - sum(samples_per_label.values())
        if adjustment > 0:
            max_weight_label = max(weights, key=weights.get)
            samples_per_label[max_weight_label] += adjustment
        selected_samples = []
        for label, num_samples in samples_per_label.items():
            if num_samples > len(report_memory[label]):
                raise ValueError(f"Not enough samples in category {label} to satisfy the weight requirement.")
            selected_samples.extend(random.sample(report_memory[label].tolist(), num_samples))
        selected_report_memory = torch.tensor(selected_samples).unsqueeze(0).to(diseaseaware_token.device)
        if selected_report_memory.shape[-1] != img_f_dim:
            report_layer = nn.Linear(selected_report_memory.shape[-1], img_f_dim)
            selected_report_memory = report_layer(selected_report_memory)

        # 记得修改
        if 'iu' in args.dataset:
            print(f"Loading LLM Qwen1.5-1.8B-Chat...")
            llama_model="Qwen1.5-0.5B-Chat"
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True, use_fast=False)
            self.llama_tokenizer.pad_token_id = 0
            self.llama_tokenizer.bos_token_id = 0
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                )
        else:
            print(f"Loading LLM {args.llm_path} ...")
            llama_model=args.llm_path
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False, trust_remote_code=True)
            self.llama_tokenizer.pad_token_id = 0
            if args.low_resource:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading Done')

        print('Loading HopfeildLayer, stored diseaseaware_token')
        self.HopfieldLayers = HopfieldLayer(
            input_size=768,
            hidden_size=1024,
            output_size=img_f_dim, # 768
            pattern_size=768,
            quantity=self.diseaseaware_token.size(1),
            scaling=args.scaling,   
            num_heads=6,             
            batch_first=True,         
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            dropout=0.1
        )
        self.HopfieldLayers.lookup_weights = nn.Parameter(self.diseaseaware_token, requires_grad=False)

        self.R_HopfieldLayers = HopfieldLayer(
            input_size=768,
            hidden_size=1024,
            output_size=img_f_dim, # 768
            pattern_size=768,
            quantity=selected_report_memory.size(1),
            scaling=args.scaling,  
            num_heads=6,              
            batch_first=True,         
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            dropout=0.1
        )
        self.R_HopfieldLayers.lookup_weights = nn.Parameter(selected_report_memory, requires_grad=False)

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


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
        if self.args.dataset == 'chinese':
            hypo = {k: [' '.join(vi) for vi in v] for k, v in hypo.items()}
            ref = {k: [' '.join(vi) for vi in v] for k, v in ref.items()}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("/wangx/home/E23201049fa/Disease/configs/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        
        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        print("load checkpoint from %s" % url_or_filename)

        return msg
    
    def load_disease_token(self, path):
        if os.path.isfile(path):  # 修正变量名
            checkpoint = torch.load(path, map_location="cpu")
            if 'model' not in checkpoint:
                raise KeyError("Checkpoint does not contain 'state_dicts'")
            checkpoint_model = checkpoint['model']
        else:
            raise RuntimeError("Checkpoint path is invalid")

        for k, v in checkpoint_model.items():
            if 'disease_tokens' in k:
                return v

        raise ValueError("Checkpoint does not contain 'disease_tokens'")

    def encode_img(self, images, segmentation=None):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image,segmentation)
            image_embeds.append(image_embed)
            
        visual_image_embeds = torch.stack(image_embeds).mean(0)
        self.image_features = visual_image_embeds
        image_embeds = self.Qformer_proj(visual_image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        disease_tokens = self.disease_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=disease_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        current_visual_tokens = query_output.last_hidden_state                          # q_fomer query output
        disease_aware_token = self.HopfieldLayers(current_visual_tokens)                # visual hopfield output
        report_memory_token = self.R_HopfieldLayers(current_visual_tokens)              # report hopfield output
        query_output_disease = self.llama_proj(self.output_proj(current_visual_tokens)) # query output
        disease_aware_token = self.llama_proj(self.output_proj(disease_aware_token))    # visual hopfield output
        report_memory_token = self.llama_proj(self.output_proj(report_memory_token))    # report hopfield output
        visual_image_embeds = self.llama_proj(visual_image_embeds)                      # original image output 

        inputs_llama = torch.cat((visual_image_embeds, query_output_disease, disease_aware_token, report_memory_token), dim=1)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama

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
        # print(p_before_embeds.size(),img_embeds.size(),p_after_embeds.size())
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        # print(wrapped_atts_img.size(),atts_img.size())
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
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
        # print(empty_targets.size(),targets.size())
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

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
        img_embeds, atts_img = self.encode_img(image)
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
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
            elif  val_score > 0.120:
                self.save_checkpoint(eval_res)

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
        img_embeds, atts_img = self.encode_img(image)
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
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
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