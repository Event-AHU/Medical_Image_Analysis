# 导入必要的库
import os
# 禁用tokenizers的并行处理以避免潜在的死锁问题
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
    MambaXrayVLCLIP模型：结合Mamba视觉编码器和CLIP对比学习的X光图像-文本多模态模型
    用于医学图像和报告文本之间的对比学习
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 保存超参数，便于模型检查点恢复
        self.save_hyperparameters(args)
        self.text_encoder_type = args.text_encoder_type

        # 加载视觉编码器
        print(f'Loading vision encoder:{args.vision_model}')
        # 根据模型类型选择base或large版本的ARM Mamba模型
        if args.type == 'base':
            self.visual_encoder = arm_base_pz16(args.type)
        else:
            self.visual_encoder = arm_large_pz16(args.type)
        
        # 如果提供了预训练模型路径，则加载预训练权重
        finetune = args.vision_model
        if finetune!='None':
            # 加载预训练检查点
            checkpoint = torch.load(finetune, map_location='cpu')
            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            checkpoint_model = checkpoint['model']

            # 处理Mamba模型的权重映射
            # 为了适配新的模型结构，需要将单个权重复制到多个分支
            new_dict = {}
            for k, v in checkpoint_model.items():
                # 处理conv1d层的权重映射
                if "conv1d" in k:
                    new_dict[k.replace("conv1d", "conv1d_b")] = v
                    new_dict[k.replace("conv1d", "conv1d_c")] = v
                    new_dict[k.replace("conv1d", "conv1d_c_b")] = v
                # 处理dt_proj层的权重映射
                if "dt_proj" in k:
                    new_dict[k.replace("dt_proj", "dt_proj_b")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
                # 处理x_proj层的权重映射
                if "x_proj" in k:
                    new_dict[k.replace("x_proj", "x_proj_b")] = v
                    new_dict[k.replace("x_proj", "x_proj_c")] = v
                    new_dict[k.replace("x_proj", "x_proj_c_b")] = v
                # 处理状态矩阵A的权重映射
                if "A" in k:
                    new_dict[k.replace("A", "A_b")] = v
                    new_dict[k.replace("A", "A_c")] = v
                    new_dict[k.replace("A", "A_c_b")] = v
                # 处理状态矩阵D的权重映射
                if "D" in k:
                    new_dict[k.replace("D", "D_b")] = v
                    new_dict[k.replace("D", "D_c")] = v
                    new_dict[k.replace("D", "D_c_b")] = v
                # 跳过解码器相关的权重
                if "dec" not in k:
                    new_dict[k] = v

            # 插值位置嵌入以适配不同的输入尺寸
            new_dict = interpolate_pos_embed(self.visual_encoder, new_dict)

            # 加载预训练模型权重（允许部分匹配）
            self.visual_encoder.load_state_dict(new_dict, strict=False)
        # 配置视觉编码器的训练策略
        if args.vis_use_lora:
            # 使用LoRA（Low-Rank Adaptation）进行参数高效微调
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,  # LoRA的秩
                                    lora_alpha=args.vis_alpha,  # LoRA的缩放因子
                                    target_modules=["query", "value"],  # 目标模块
                                    lora_dropout=args.lora_dropout,  # dropout率
                                    bias="none",  # 不训练bias
                                    modules_to_save=["classifier"],  # 需要保存的模块
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            # 冻结视觉编码器的所有参数
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            # 视觉编码器所有参数可训练
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        # 加载文本编码器
        print(f"Loading text encoder : {self.text_encoder_type}...")
        if self.text_encoder_type == 'Bio_ClinicalBERT':  
            # 使用医学领域预训练的BERT模型
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            # 如果没有BOS token，使用CLS token作为替代
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
            self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")    

        # 设置投影维度和投影层
        self.projection_dim = args.projection_dim
        # 视觉特征投影层：将视觉特征投影到共同的嵌入空间
        self.vision_proj = nn.Linear(self.visual_encoder.num_features, self.projection_dim)
        # 文本特征投影层：将文本特征投影到共同的嵌入空间
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim)
        
        # CLIP对比学习的温度参数
        self.temperature = 0.07
        # 可学习的logit缩放参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))

        # 用于跟踪最小损失值，用于模型保存
        self.min_loss = -1

        # 如果提供了增量文件（delta file），加载预训练的模型权重
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def encode_img(self, images):
        """
        编码图像特征
        Args:
            images: 输入图像列表
        Returns:
            image_embeds: 投影后的图像嵌入向量
        """
        image_embeds = []
        # 对每张图像进行编码
        for image in images:
            device = image.device
            # 通过视觉编码器提取图像特征
            image_embed = self.visual_encoder(image)
            image_embeds.append(image_embed)
        
        # 将多张图像的特征进行平均池化
        image_embeds = torch.stack(image_embeds).mean(0)
        # 对序列维度进行平均池化，得到全局图像表示
        image_embeds = image_embeds.mean(dim=1)
        # 通过投影层映射到共同的嵌入空间
        image_embeds = self.vision_proj(image_embeds)
        return image_embeds

    def encode_txt(self, text_tokens):
        """
        编码文本特征
        Args:
            text_tokens: 分词后的文本tokens
        Returns:
            text_features: 投影后的文本嵌入向量
        """
        if self.text_encoder_type == 'Bio_ClinicalBERT':
            # 通过BERT编码器获取文本的隐藏状态
            text_features = self.text_encoder(text_tokens['input_ids'], attention_mask = text_tokens['attention_mask'])["last_hidden_state"]
        
        # 获取每个序列的最后一个有效token的位置（EOS token位置）
        eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
        # 提取每个序列最后一个有效token的特征作为句子表示
        text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        # 通过投影层映射到共同的嵌入空间
        text_features = self.text_proj(text_features)
        return text_features

    def forward(self, samples):
        """
        前向传播函数
        Args:
            samples: 包含图像和文本的样本字典
        Returns:
            包含损失值的字典
        """
        # 提取图像和文本数据
        image = samples["image"]
        report = samples["input_text"]
        
        # 对文本进行分词和编码
        text_tokens = self.tokenizer(report, padding="max_length", truncation=True, return_tensors="pt", max_length=128).to(image[0].device)
        
        # 编码图像和文本特征
        image_features = self.encode_img(image)
        text_features = self.encode_txt(text_tokens)
        
        # 对特征进行L2归一化，这是CLIP的关键步骤
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度作为logits
        logit_scale = self.logit_scale.exp()  # 将log scale转换为实际的scale值
        logits_per_image = logit_scale * image_features @ text_features.t()  # 图像到文本的相似度矩阵
        logits_per_text = logits_per_image.t()  # 文本到图像的相似度矩阵
        
        # 创建标签：对角线元素为正样本对
        labels = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=logits_per_image.device)

        # 计算对比学习损失：图像到文本 + 文本到图像的交叉熵损失
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        Args:
            batch: 训练批次数据
            batch_idx: 批次索引
        Returns:
            训练结果字典
        """
        result = self(batch)
        # 记录训练指标到进度条
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, loss):
        """
        保存模型检查点
        Args:
            loss: 当前损失值
        """
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        
        # 获取所有需要梯度的参数
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        
        # 获取模型状态字典，只保存可训练的参数
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        
        # 构建保存对象
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        
        # 创建检查点目录
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        
        # 构建保存路径
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_loss{:3f}.pth".format(current_epoch, global_step, loss),
        )
        
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    def on_train_epoch_end(self):
        """
        训练epoch结束时的回调函数
        根据损失值决定是否保存模型检查点
        """
        avg_loss = self.trainer.callback_metrics["loss"]
        
        # 如果是第一个epoch或者当前损失小于历史最小损失，则保存检查点
        if self.min_loss == -1 :
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
        elif avg_loss < self.min_loss and self.min_loss != -1:
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        Returns:
            包含优化器和调度器的字典
        """
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        """
        自定义进度条显示内容
        Returns:
            进度条字典，移除版本号显示
        """
        # 不显示版本号
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """
        清零优化器梯度
        Args:
            epoch: 当前epoch
            batch_idx: 批次索引
            optimizer: 优化器对象
        """
        optimizer.zero_grad()