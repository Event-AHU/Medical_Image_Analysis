import os
import json
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType

from dataset.data_helper import FieldParser
from einops import rearrange,reduce
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" 


class EncoderProjectorQFormer(nn.Module):
    def __init__(self, downsample_rate, encoder_dim, llm_dim, ffn_dim: int = 2048, **kwargs):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 2
        
        self.query_len = 64
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.negative_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)
        
        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
    
    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj
class R2GenCSR(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.demo:
            None
        else:
            self.save_hyperparameters(args)
        # self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.proj = args.proj
        self.chosen= args.chosen
        self.llm = args.llm

        if self.chosen != 'vmamba':
            self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        else:
 
            from VMamba.classification.config import get_config
            from VMamba.classification.models import build_model
            from collections import OrderedDict
            class vmamba_args:
                cfg ='./VMamba/classification/configs/vssm1/vssm_base_224.yaml'
                opts= None
                batch_size=8
                zip=None
                cache_mode='part'
                pretrained = args.vision_model
                resume =None
                accumulation_steps =None
                use_checkpoint=None
                disable_amp=None
                output ='./output'
                tag =None
                throughput=None
                traincost =None
                fused_layernorm=None
                optim =None
                model_ema =True
                model_ema_decay =0.9999
                model_ema_force_cpu =False
                memory_limit_rate = -1
            config = get_config(vmamba_args)
            self.visual_encoder = build_model(config,is_pretrain=True)
            
            checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
            if 'model' in checkpoint:
                msg = self.visual_encoder.load_state_dict(checkpoint['model'], strict=False)
                # self.visual_encoder = self.visual_encoder.to(torch.bfloat16)
                print("=> loaded pretrinaed vssm model successfully")


            
            
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
        
        print('Loading LLAMA')
        print(self.llm)
        if self.llm != 'llama2':
            self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model,)
            # self.llama_tokenizer.pad_token_id = 0
            self.llama_tokenizer.pad_token_id = 0
             
            self.llama_tokenizer.bos_token_id = 0
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype="auto",
                device_map="auto"
                )
        else:
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
            # llama
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False,target_modules=["q_proj", "v_proj"], r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            ########### 
            if self.args.llm_freeze is True:
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
            print('Loading LLAMA Done')
        if self.proj != 'qformer':
            self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        else:
            self.llama_proj = EncoderProjectorQFormer(0,encoder_dim = self.visual_encoder.num_features,llm_dim = self.llama_model.config.hidden_size)

        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = self.args.instruction
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


        self.context_base = []
        if self.args.context_pair >0:
            self.negative_samples,self.positive_samples = self.context_sample(self.args.context_pair)
        else:
            self.negative_samples,self.positive_samples = None,None
        
    
    
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


    
    def encode_img(self, images,global_only=False,global_only_return=False,use_feature_mean=True,featuremap_folder=None):
        image_embeds = []
        for image in images:
            device = image.device
            if self.chosen == 'vmamba':
                if global_only:
                    image_embed = self.visual_encoder(image,global_only)
                else:
                    image_embed = self.visual_encoder(image,global_only,featuremap_folder=featuremap_folder)
                    image_embed = rearrange(image_embed, 'b h w e -> b (h w) e')
            elif self.chosen == 'vim':
                image_embed = self.visual_encoder(image,return_features=True)
            else:
                if global_only:
                    image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
                    image_embed = reduce(image_embed, 'b l e -> b e','mean')
                else:
                    image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
        if self.args.use_feature_mean or global_only is True:
            image_embeds = torch.stack(image_embeds).mean(0)
            if global_only_return:
                return image_embeds,None
        else:
            if len(image_embeds)==1 :
                #for Test and Val with the same token-len as Train
                image_embeds = image_embeds + image_embeds
            image_embeds = torch.concat(image_embeds,dim=1)
        # image_embeds = torch.stack(image_embeds)
        if self.proj != 'qformer':
            inputs_llama = self.llama_proj(image_embeds)
        else:
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            inputs_llama = self.llama_proj.forward(image_embeds,image_attention_mask)
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
    
    def image_prompt_wrap(self, img_embeds, atts_img,prompt):
        # prompt=f'Note: <Img><ImageHere></Img> with desease'
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
    
    def context_dict2datalorderFormat(self,batch_to_return):
        batch = {}
        batch["id"] = [item['id'] for item in batch_to_return]
        batch["input_text"] = [item['input_text'] for item in batch_to_return]
        batch["image"] = torch.stack([item['image'][0]  for item in batch_to_return]).to(torch.bfloat16)
        
        return batch
    
    def context_sample(self, num=3):
        parser = FieldParser(self.args)
        # meta = json.load(open(self.args.annotation, 'r'))
        meta = json.loads(open(self.args.annotation,  'r', encoding='utf-8').read())
        df = pd.DataFrame(meta['train'])


        if self.args.dataset == 'chinese':
            negative_tag = '未见'
            negative = df[df['impressions'].str.contains(negative_tag)]
            positive = df[~df['impressions'].str.contains(negative_tag)]

  
        elif self.args.dataset =='mimic_cxr':
            if self.args.context_retrieval_mode == 'chexbert':
                
                ann_context = pd.read_csv('../_dataset/mimic_cxr_pretrain/ann_chexbert.csv')
                ann_context['image_path'] = ann_context['image_path'].apply(lambda x: eval(x))
                negative = ann_context[ann_context['no_finding']==1]
                positive = ann_context[ann_context['no_finding']!=1]
            elif self.args.context_retrieval_mode == 'random':
                
                negative = df.sample(60,random_state=self.args.context_pair_seed)
                positive = df.sample(60,random_state=self.args.context_pair_seed)
            else:
                positive_tag = 'note'
                negative = df[~df['report'].str.contains(positive_tag)]
                positive = df[df['report'].str.contains(positive_tag)]
                
        elif self.args.dataset =='iu_xray':
            if self.args.context_retrieval_mode == 'chexbert':
                
                ann_context = pd.read_csv('../_dataset/iu_xray/ann_chexbert.csv')
                ann_context['image_path'] = ann_context['image_path'].apply(lambda x: eval(x))
                negative = ann_context[ann_context['no_finding']==1]
                positive = ann_context[ann_context['no_finding']!=1]
            elif self.args.context_retrieval_mode == 'random':
                
                negative = df.sample(60,random_state=self.args.context_pair_seed)
                positive = df.sample(60,random_state=self.args.context_pair_seed)
            else:
                positive_tag = 'note'
                negative = df[~df['report'].str.contains(positive_tag)]
                positive = df[df['report'].str.contains(positive_tag)]
        
        negative_samples = negative.sample(30,random_state=self.args.context_pair_seed)
        negative_samples=negative_samples.to_dict('records')
        # print(negative_samples)
        positive_samples = positive.sample(30,random_state=self.args.context_pair_seed)
        positive_samples=positive_samples.to_dict('records')
        # print(positive_samples)

        negative_samples = negative_samples[:num]
        positive_samples = positive_samples[:num]
        print('negative_samples:',[ dic['id'] for dic in negative_samples])
        print('positive_samples:',[ dic['id'] for dic in positive_samples])

        negative_samples = [parser.transform_with_parse(dic) for dic in negative_samples]
        positive_samples = [parser.transform_with_parse(dic) for dic in positive_samples]

        negative_samples = self.context_dict2datalorderFormat(negative_samples)
        positive_samples = self.context_dict2datalorderFormat(positive_samples)

        return negative_samples,positive_samples

    def context_encode_with_wrap(self,image,img_embeds,global_contex = True):
        with torch.no_grad():
            if self.args.context_fixed:
                negative_samples,positive_samples = self.negative_samples,self.positive_samples
                
            else:
                self.args.context_pair_seed = None
                negative_samples,positive_samples = self.context_sample()
            # before_proj_res = True
            # print('before_proj_res',before_proj_res)
            # before_proj_res = False
            global_img_embeds, global_atts_img = self.encode_img(image,global_only=True,global_only_return=self.args.before_proj_res)
            
            positive_img_embeds, positive_atts_img = self.encode_img([positive_samples["image"].to(img_embeds.device)],global_only=global_contex,global_only_return=self.args.before_proj_res)
            negative_img_embeds, negative_atts_img = self.encode_img([negative_samples["image"].to(img_embeds.device)],global_only=global_contex,global_only_return=self.args.before_proj_res)

            if self.args.before_proj_res:
                global_img_embeds = rearrange(global_img_embeds,'b e -> b 1 e').expand(-1, positive_img_embeds.size(0), -1)
                residual_neg_embeds = global_img_embeds -positive_img_embeds
                residual_pos_embeds = global_img_embeds -negative_img_embeds
                if self.proj != 'qformer':
                    positive_img_embeds = self.llama_proj(residual_neg_embeds)
                    negative_img_embeds = self.llama_proj(residual_pos_embeds)
                else:
                    image_attention_mask = torch.ones(positive_img_embeds.size()[:-1], dtype=torch.long, device=positive_img_embeds.device)
                    positive_img_embeds = self.llama_proj.forward(positive_img_embeds,image_attention_mask)

                    image_attention_mask = torch.ones(negative_img_embeds.size()[:-1], dtype=torch.long, device=negative_img_embeds.device)
                    negative_img_embeds = self.llama_proj.forward(negative_img_embeds,image_attention_mask)

                positive_atts_img = torch.ones(positive_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
                negative_atts_img = torch.ones(negative_img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

                positive_img_prompt = self.args.positive
                negative_img_prompt = self.args.negative
                positive_img_embeds_wrap, positive_atts_img_wrap = self.image_prompt_wrap(positive_img_embeds,positive_atts_img,prompt=positive_img_prompt)
                negative_img_embeds_wrap, negative_atts_img_wrap = self.image_prompt_wrap(negative_img_embeds,negative_atts_img,prompt=negative_img_prompt)


                context_embeds = torch.concat((negative_img_embeds_wrap,positive_img_embeds_wrap),dim=1)
                context_atts_img = torch.concat((negative_atts_img_wrap,positive_atts_img_wrap),dim=1)
                residual_img_embeds = context_embeds
                residual_atts_img = context_atts_img
                return residual_img_embeds,residual_atts_img
            
            elif self.args.after_proj_res_visual_only:
                global_img_embeds = rearrange(global_img_embeds,'b e -> b 1 e').expand(-1, positive_img_embeds.size(0), -1)
                positive_img_embeds = global_img_embeds -positive_img_embeds
                negative_img_embeds = global_img_embeds -negative_img_embeds

                # positive_img_embeds = rearrange(positive_img_embeds,'b e -> b 1 e')
                # negative_img_embeds = rearrange(negative_img_embeds,'b e -> b 1 e')
                positive_atts_img = rearrange(positive_atts_img,'b -> b 1 ')
                negative_atts_img = rearrange(negative_atts_img,'b -> b 1 ')

                positive_img_prompt = self.args.positive
                negative_img_prompt = self.args.negative
                positive_img_embeds_wrap, positive_atts_img_wrap = self.image_prompt_wrap(positive_img_embeds,positive_atts_img,prompt=positive_img_prompt)
                negative_img_embeds_wrap, negative_atts_img_wrap = self.image_prompt_wrap(negative_img_embeds,negative_atts_img,prompt=negative_img_prompt)

                context_embeds = torch.concat((negative_img_embeds_wrap,positive_img_embeds_wrap),dim=1)
                context_atts_img = torch.concat((negative_atts_img_wrap,positive_atts_img_wrap),dim=1)
                residual_img_embeds = context_embeds
                residual_atts_img = context_atts_img
                return residual_img_embeds,residual_atts_img[:1].expand(positive_img_embeds.size(0), -1)

            
            if global_contex:
                # context_embeds (context batch, emb)
                positive_img_embeds = rearrange(positive_img_embeds,'b e -> b 1 e')
                negative_img_embeds = rearrange(negative_img_embeds,'b e -> b 1 e')
                positive_atts_img = rearrange(positive_atts_img,'b -> b 1 ')
                negative_atts_img = rearrange(negative_atts_img,'b -> b 1 ')
            else:
                # context_embeds (context batch, len, emb)
                positive_img_embeds = self.layer_norm(positive_img_embeds)
                negative_img_embeds = self.layer_norm(negative_img_embeds)



            # get context prompt
            positive_img_prompt = self.args.positive  # positive_img by keyword
            negative_img_prompt = self.args.negative

            # wrap and project
            positive_img_embeds_wrap, positive_atts_img_wrap = self.image_prompt_wrap(positive_img_embeds,positive_atts_img,prompt=positive_img_prompt)
            negative_img_embeds_wrap, negative_atts_img_wrap = self.image_prompt_wrap(negative_img_embeds,negative_atts_img,prompt=negative_img_prompt)
            
            # context embedding -> [negtive,positive] 
            context_embeds = torch.concat((negative_img_embeds_wrap,positive_img_embeds_wrap),dim=1)
            context_atts_img = torch.concat((negative_atts_img_wrap,positive_atts_img_wrap),dim=1)

            context_embeds = rearrange(context_embeds,'b l e -> 1 (b l) e').expand(img_embeds.size(0),-1,  -1) 
            context_atts_img =  rearrange(context_atts_img,'b l -> 1 (b l)').expand(img_embeds.size(0),-1)
            global_img_embeds = rearrange(global_img_embeds,'b e -> b 1 e').expand(-1, context_embeds.size(1), -1)
            
            # context residual          size: (batch, batch_context, len,emb)
            residual_img_embeds = global_img_embeds - context_embeds
            residual_atts_img = context_atts_img

        return residual_img_embeds,residual_atts_img

        # img_embeds = torch.concat((residual_img_embeds,img_embeds),dim=1)
        # atts_img = torch.concat((context_atts_img,atts_img),dim=1)

        # return img_embeds,atts_img

    
    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
        if self.args.context_pair>0:
            context_embeds,contex_atts_img = self.context_encode_with_wrap(image,img_embeds,global_contex = True)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            img_embeds = torch.concat((context_embeds,img_embeds),dim=1)
            atts_img = torch.concat((contex_atts_img,atts_img),dim=1)
        else:
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)



        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.agrs.max_length,
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
            "config": self.agrs,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.agrs.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.agrs.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            # padding=True,
            padding="max_length",
            truncation=True,
            max_length=self.agrs.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        #########
        if self.args.context_pair>0:
            context_embeds,contex_atts_img = self.context_encode_with_wrap(image,img_embeds,global_contex = True)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            img_embeds = torch.concat((context_embeds,img_embeds),dim=1)
            atts_img = torch.concat((contex_atts_img,atts_img),dim=1)
        else:
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
            # attention_mask = attention_mask,######################################
            num_beams=self.agrs.beam_size,
            do_sample=self.agrs.do_sample,
            min_new_tokens=self.agrs.min_new_tokens,
            max_new_tokens=self.agrs.max_new_tokens,
            repetition_penalty=self.agrs.repetition_penalty,
            length_penalty=self.agrs.length_penalty,
            temperature=self.agrs.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        # ref = samples['input_text']

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
        output_text = output_text.replace('!', '') # some model use "!" as special token, remove it
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

        result_folder = os.path.join(self.agrs.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.agrs.scorer_types, self.agrs.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            self.save_checkpoint(eval_res)
            if val_score > self.val_score:
                # self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def demo_test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        #########
        if self.args.context_pair>0:
            context_embeds,contex_atts_img = self.context_encode_with_wrap(image,img_embeds,global_contex = True)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            img_embeds = torch.concat((context_embeds,img_embeds),dim=1)
            atts_img = torch.concat((contex_atts_img,atts_img),dim=1)
        else:
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
            # attention_mask = attention_mask,######################################
            num_beams=self.args.beam_size,
            do_sample=self.args.do_sample,
            min_new_tokens=self.args.min_new_tokens,
            max_new_tokens=self.args.max_new_tokens,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            temperature=self.args.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.agrs.max_length,
            add_special_tokens=False
        )
        image = samples["image"]
        ####
        featuremap_folder=None
        if self.args.featuremap:
            featuremap_folder = './featuremap/'+str(samples["id"][0])
            os.system(f'mkdir -p {featuremap_folder}')
        img_embeds, atts_img = self.encode_img(image,featuremap_folder=featuremap_folder)
        img_embeds = self.layer_norm(img_embeds)
        # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        #########
        if self.args.context_pair>0:
            context_embeds,contex_atts_img = self.context_encode_with_wrap(image,img_embeds,global_contex = True)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            img_embeds = torch.concat((context_embeds,img_embeds),dim=1)
            atts_img = torch.concat((contex_atts_img,atts_img),dim=1)
        else:
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
            # attention_mask = attention_mask,
            num_beams=self.agrs.beam_size,
            do_sample=self.agrs.do_sample,
            min_new_tokens=self.agrs.min_new_tokens,
            max_new_tokens=self.agrs.max_new_tokens,
            repetition_penalty=self.agrs.repetition_penalty,
            length_penalty=self.agrs.length_penalty,
            temperature=self.agrs.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        if self.args.featuremap:
            file_path = f"{featuremap_folder}/predict.txt"
            txt = 'hypo: '.join(hypo)
            txt = txt+ '\n'+'ref: '.join(ref)+ '\n'
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(txt)


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

        result_folder = os.path.join(self.agrs.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        self.print(f"Test result of {self.agrs.delta_file}: {eval_res}")

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.agrs.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.agrs.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()