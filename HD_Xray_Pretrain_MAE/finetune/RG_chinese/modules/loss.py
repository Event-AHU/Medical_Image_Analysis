import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def __init__(self, visual_token_num, text_token_num):
        super(LanguageModelCriterion, self).__init__()
        # self.fc_pre = nn.Linear(visual_token_num, text_token_num)
        
    def _forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        
    def forward(self, predictions, labels, target_mask):
        # breakpoint()
        # predictions torch.Size([1, 77, 21129])
        # labels torch.Size([1, 77])
        # target_mask torch.Size([1, 77])
        predictions = predictions.view(-1, predictions.size(-1))  #torch.Size([77, 21129])
        labels = labels.contiguous().view(-1) # torch.Size([77])
        target_mask = target_mask.contiguous().view(-1).float() # torch.Size([77])
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过 mask 取消 pad 和句子 a 部分预测的影响
    
    # def forward(self, predictions, labels, target_mask):
    #     # print("predictions--->", predictions.shape)
    #     # print("labels--->", labels.shape)
    #     # print("target_mask--->", target_mask.shape)
    #     # device = predictions.device
    #     # predictions_ = torch.zeros(predictions.shape[0], labels.shape[1]).to(device)
    #     # predictions_[:,:predictions.shape[1]]  =  predictions
    #     # predictions  =predictions_
    #     # predictions = predictions.view(-1, predictions.shape[2], predictions.shape[3])
    #     # print("predictions reshape--->", predictions.shape)
    #     predictions = predictions.transpose(1, 2)
    #     # self.fc_pre.to(device=predictions.device)
    #     # print('self.fc_pre.device', self.fc_pre.device)
    #     # print('predictions.device', predictions.device)
    #     # print('labels.device', labels.device)
    #     self.fc_pre.to(device=predictions.device)
    #     predictions = self.fc_pre(predictions)
    #     predictions = predictions.transpose(1, 2)
    #     # print("predictions fc--->", predictions.shape)
    #     predictions = predictions.contiguous()
    #     predictions = predictions.view(-1, predictions.size(-1)).float()

    #     # print("predictions final--->", predictions)

    #     # labels---> torch.Size([32, 96])
    #     # target_mask---> torch.Size([32, 96])
    #     # predictions new---> torch.Size([32, 96])
        
    #     labels = labels.contiguous().view(-1).float()
    #     target_mask = target_mask.contiguous().view(-1).float()
    #     # print("labels new--->", labels.shape)
        
    #     # print('predictions.dtype',type(predictions))
    #     # print('labels.dtype',type(labels))
    #     pred = predictions.to(torch.float)
    #     target = labels.to(torch.long)
        
        
    #     loss = nn.CrossEntropyLoss()
    #     return (loss(pred, target) * target_mask).sum() / target_mask.sum() ## 通过 mask 取消 pad 和句子 a 部分预测的影响


def compute_loss(output, reports_ids, reports_masks):
    visual_token_num = output.shape[1]
    text_token_num = reports_ids[:, 1:].shape[1]
    criterion = LanguageModelCriterion(visual_token_num, text_token_num)
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])    # .mean()
    return loss
