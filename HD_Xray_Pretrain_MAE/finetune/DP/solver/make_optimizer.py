import torch


def make_optimizer(model, lr=8e-3, weight_decay=1e-4, momentum=0.9):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'SGD')(params, momentum=momentum)
    
    return optimizer
