import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import functional as F
from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16
from arm.Finetuning.util.pos_embed import interpolate_pos_embed

class MambaXrayVLCLIP(pl.LightningModule):
    """
    MambaXrayVLCLIP model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.text_encoder_type = args.text_encoder_type

        print(f'Loading vision encoder:{args.vision_model}')
        # print(f'you choose the vison encoder model Mamba {self.type}')
        if args.type == 'base':
            self.visual_encoder = arm_base_pz16(args.type)
        else:
            self.visual_encoder = arm_large_pz16(args.type)
        finetune = args.vision_model
        if finetune!='None':
            checkpoint = torch.load(finetune, map_location='cpu')
            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            checkpoint_model = checkpoint['model']

            new_dict = {}
            for k, v in checkpoint_model.items():
                if "conv1d" in k:
                    new_dict[k.replace("conv1d", "conv1d_b")] = v
                    new_dict[k.replace("conv1d", "conv1d_c")] = v
                    new_dict[k.replace("conv1d", "conv1d_c_b")] = v
                if "dt_proj" in k:
                    new_dict[k.replace("dt_proj", "dt_proj_b")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
                if "x_proj" in k:
                    new_dict[k.replace("x_proj", "x_proj_b")] = v
                    new_dict[k.replace("x_proj", "x_proj_c")] = v
                    new_dict[k.replace("x_proj", "x_proj_c_b")] = v
                if "A" in k:
                    new_dict[k.replace("A", "A_b")] = v
                    new_dict[k.replace("A", "A_c")] = v
                    new_dict[k.replace("A", "A_c_b")] = v
                if "D" in k:
                    new_dict[k.replace("D", "D_b")] = v
                    new_dict[k.replace("D", "D_c")] = v
                    new_dict[k.replace("D", "D_c_b")] = v
                if "dec" not in k:
                    new_dict[k] = v

            # interpolate position embedding
            new_dict = interpolate_pos_embed(self.visual_encoder, new_dict)
            print(new_dict)
            # load pre-trained model
            self.visual_encoder.load_state_dict(new_dict, strict=False)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    #target_modules=["query", "value"],
                                    arget_modules=["conv1d_b", "conv1d_c"],
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

        print(f"Loading text encoder : {self.text_encoder_type}...")
        if self.text_encoder_type == 'Bio_ClinicalBERT':  
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
            self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")    

        self.projection_dim = args.projection_dim
        self.vision_proj = nn.Linear(self.visual_encoder.num_features, self.projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim)
        self.temperature = 0.07 # 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))

        self.min_loss = -1

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        image_embeds = image_embeds.mean(dim=1)
        image_embeds = self.vision_proj(image_embeds)
        return image_embeds

    def encode_txt(self, text_tokens):
        if self.text_encoder_type == 'Bio_ClinicalBERT':
            text_features = self.text_encoder(text_tokens['input_ids'], attention_mask = text_tokens['attention_mask'])["last_hidden_state"]
        eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
        text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        text_features = self.text_proj(text_features)
        return text_features

    def forward(self, samples):
        image = samples["image"]
        report = samples["input_text"]
        text_tokens = self.tokenizer(report, padding="max_length", truncation=True, return_tensors="pt", max_length=128).to(image[0].device)
        image_features = self.encode_img(image)
        text_features = self.encode_txt(text_tokens)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # breakpoint()
        labels = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=logits_per_image.device)

        # loss = self.cliploss(image_features, text_features, self.logit_scale.exp())["contrastive_loss"]
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, loss):
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
            "checkpoint_epoch{}_step{}_loss{:3f}.pth".format(current_epoch, global_step, loss),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["loss"]
        if self.min_loss == -1 :
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
        elif avg_loss < self.min_loss and self.min_loss != -1:
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
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