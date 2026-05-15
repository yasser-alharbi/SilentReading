import torch
from torch.optim import AdamW

def build_optimizer(args, model, mode="cet-mae"):
    parameters = [p for p in model.parameters() if p.requires_grad]

    if mode == "cet-mae":
        lr = float(args.get('cet_mae_lr', 5e-7))
        weight_decay = float(args.get('weight_decay', 1e-2))
        beta1 = float(args.get('adam_beta1', 0.9))
        beta2 = float(args.get('adam_beta2', 0.999))

        optimizer = AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
    else:
        optimizer = AdamW(parameters, lr=1e-4)

    return optimizer