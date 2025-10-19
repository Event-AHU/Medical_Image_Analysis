import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from lightning_tools.optim import config_optimizer

from peft import get_peft_model, LoraConfig, TaskType
#from peft import LoraConfig

import math
from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16
import types
from models.hybrid_decoder_layer import Qwen2HybridDecoderLayer
'''
from mamba_peft.src.peft.tuners import (
   # LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    MambaPEFTConfig,
    #get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)'''


class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32, use_act=False,dropout=0.1):
        super().__init__()

        self.adapter_down = nn.Linear(in_channels, dim, bias=False)
        self.adapter_up = nn.Linear(dim, out_channels, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))

        self.act = nn.GELU() if use_act else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.adapter_up(self.act(self.adapter_down(x)))


class MambaXrayVLDownStream(pl.LightningModule):
    """
    MambaXrayVLDownStream model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.json_path = args.annotation
        self.type = args.type
        # 添加hybrid decoder相关配置
        self.use_hybrid_decoder = getattr(args, 'use_hybrid_decoder', False)
        self.cross_attn_every_n_layers = getattr(args, 'cross_attn_every_n_layers', 4)
        self.cross_attn_implementation = getattr(args, 'cross_attn_implementation', 'text-only-vanilla')
        self.cross_attn_gating_type = getattr(args, 'cross_attn_gating_type', 'channel-wise-dynamic-sigmoid')
        self.spatial_stride = getattr(args, 'spatial_stride', 2)

        self.use_lora_X = getattr(args, 'lora_X', False)
        self.dim_X = getattr(args, 'dim_X', 64)
        self.s_X = getattr(args, 's_X', 1.0)

        print(f'Loading vision encoder:{args.vision_model}')
        if 'B' in args.vision_model:
            self.visual_encoder = arm_base_pz16(self.type)
        else:
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
           

        if self.use_lora_X:
            self._apply_lora_X_to_model(self.visual_encoder)


        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    #target_modules=["query", "value"],
                                    target_modules=["x_proj"],
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
        # breakpoint()


        print(f'Loading LLM ...')

       
        llama_model="meta-llama/Llama-2-7b-chat-hf"
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                )
        if args.llm_use_lora:
                self.embed_tokens = self.llama_model.get_input_embeddings()
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=args.llm_r, 
                    lora_alpha=args.llm_alpha, 
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "v_proj"],  # 需要插入LoRA层的模块
                )
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                self.llama_model.print_trainable_parameters()
                llama_dict = {}
                for k, v in checkpoint_model.items():
                    if "text_encoder." in k:
                        llama_dict[k.replace("text_encoder.", "")] = v
                self.llama_model.load_state_dict(llama_dict, strict = False)
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                print('Loading LLAMA LoRA Done')         
        else:
                self.embed_tokens = self.llama_model.get_input_embeddings()
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                print('Loading LLAMA Done') 
                
        # 初始化hybrid decoder layers
        if self.use_hybrid_decoder:
            self._initialize_hybrid_decoder_layers()

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

    def _initialize_hybrid_decoder_layers(self):
        print("Initializing hybrid decoder layers...")

        if hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'layers'):
            layers = self.llama_model.model.layers
        elif hasattr(self.llama_model, 'layers'):
            layers = self.llama_model.layers
        else:
            print("Warning: Could not find decoder layers in the model")
            return

        num_layers = len(layers)

        hybrid_layer_idx = []
        for i in range(0, num_layers, self.cross_attn_every_n_layers):
            hybrid_layer_idx.append(i)

        print(f"Replacing layers {hybrid_layer_idx} with hybrid decoder layers")

        for idx in hybrid_layer_idx:
             if idx < len(layers):
                original_layer = layers[idx]
                hybrid_layer = Qwen2HybridDecoderLayer(
                     self.llama_model.config,
                     layer_idx=idx,
                     is_hyper_enabled=True,
                     cross_attn_implementation=self.cross_attn_implementation,
                     cross_attn_gating_type=self.cross_attn_gating_type
                )
                #加载原始层的权重
                hybrid_layer.load_state_dict(original_layer.state_dict(), strict=False)
                layers[idx] = hybrid_layer
        print("Hybrid decoder layers initialized")

    def split_slow_fast_tokens(self, visual_tokens, spatial_stride=2):
        if isinstance(visual_tokens, torch.Tensor):
            b, n, c = visual_tokens.shape
            h = w = int(n ** 0.5)
            fast_visual_tokens = nn.functional.avg_pool2d(
                visual_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2),
                kernel_size=spatial_stride,
                stride=spatial_stride
            ).flatten(2, 3).transpose(1, 2)
            slow_visual_tokens = visual_tokens

            return fast_visual_tokens, slow_visual_tokens
        else:
            fast_tokens = []
            slow_tokens = []
            for token in visual_tokens:
                fast, slow = self.split_slow_fast_tokens(token, spatial_stride)
                fast_tokens.append(fast)
                slow_tokens.append(slow)
            return fast_tokens, slow_tokens


    def clear_hybrid_layers(self):
        if not self.use_hybrid_decoder:
            return

        if hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'layers'):
            layers = self.llama_model.model.layers
        elif hasattr(self.llama_model, 'layers'):
            layers = self.llama_model.layers
        else:
            return

        for layer in layers:
            if hasattr(layer, 'clear_vis_x'):
                layer.clear_vis_x()

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            ##(Meteor(), "METEOR"),
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

    def _apply_lora_X_to_model(self, model):
        for name, module in model.named_modules():
            if hasattr(module, 'in_proj') and hasattr(module, 'out_proj'):
                if hasattr(module, 'hidden_size') and hasattr(module, 'intermediate_size'):
                    in_ch, out_ch = module.hidden_size, module.intermediate_size
                elif hasattr(module, 'd_model') and hasattr(module, 'd_inner'):
                    in_ch, out_ch = module.d_model, module.d_inner
                else:
                    continue

                module.lora_X = Adapter(in_ch, out_ch // 2, self.dim_X)
                module.s_X = self.s_X

                original_forward = module.forward
                def new_forward(self_module, *args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    if hasattr(self_module, 'lora_X') and self_module.lora_X is not None:

                        hidden_states = args[0]
                        lora_output = self_module.s_X * self_module.lora_X(hidden_states)
                        if isinstance(output, tuple):
                            output_list = list(output)
                            if len(output_list) > 0:
                                if isinstance(output_list[0], torch.Tensor):
                                    
                                    half_size = output_list[0].shape[-1] // 2
                                    output_list[0][..., :half_size] = output_list[0][..., :half_size] + lora_output
                            output = tuple(output_list)
                        elif isinstance(output, torch.Tensor):                         
                            half_size = output.shape[-1] // 2
                            output[..., :half_size] = output[..., :half_size] + lora_output
                    return output


                module.forward = types.MethodType(new_forward, module)
    def encode_img(self, images, segmentation=None):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image,segmentation)
            image_embeds.append(image_embed)

        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
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
        # samples
        # 'id': ['CXR2279_IM-0865']
        # 'input_text':
        # ['heart size is enlarged . the aorta is unfolded . otherwise the mediastinal contour is... intraperitoneal air under the diaphragm .']
        # 'image':
        # [tensor([[[[-1.0733, -1.0904, -1.0904,  ..., -1.6555, -1.6384, -1.6384],
        #           [-1...7751,  0.6879]]]],
        #        device='cuda:0'), tensor([[[[-1.3644, -1.3644, -1.3815,  ..., -0.3883, -0.4397, -0.5082],
        #           [-1...5256, -1.4384]]]],
        #        device='cuda:0')]

        # if self.args.dataset == 'chinese':
        #     hypo = [' '.join(tttt) for tttt in hypo]
        #     ref = [' '.join(tttt) for tttt in ref]
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
            elif  val_score > 0.125:
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
