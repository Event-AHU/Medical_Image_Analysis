import torch
import torch.nn as nn
import numpy as np

from timm.models.layers import trunc_normal_
from modules.visual_extractor import VisualExtractor, vit_base_patch16, vit_large_patch16
from modules.encoder_decoder import EncoderDecoder
from modules.pos_embed import interpolate_pos_embed
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
         
        if args.mae_pretrained:
            # self.visual_extractor = vit_base_patch16(global_pool=args.global_pool, img_size=self.args.img_size)
            self.visual_extractor = vit_large_patch16(global_pool=args.global_pool, img_size=self.args.img_size)
            checkpoint = torch.load(args.mae_pretrained, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.mae_pretrained)
            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.visual_extractor.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            
            # interpolate position embedding
            interpolate_pos_embed(self.visual_extractor, checkpoint_model)
            
            # load pre-trained model
            msg = self.visual_extractor.load_state_dict(checkpoint_model, strict=False)
            
            
        else:
            self.visual_extractor = VisualExtractor(args) 
        
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])      # [B, 49, dim]  [B, dim]
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

