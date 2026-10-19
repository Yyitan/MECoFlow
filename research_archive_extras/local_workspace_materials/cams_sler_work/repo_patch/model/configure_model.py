import torch
from model.cams import CAMS

def configure_model(args,dataset):
    
    cams = CAMS(args,dataset)
    keywords = ['csp','norm','tr_','adapter','proj_att','proj_obj','proj_com','temp_logit','cross_attention','latent_units','sler_']

    freeze_legacy = getattr(args, 'sler_freeze_legacy', False)
    train_scope = getattr(args, 'train_scope', 'legacy')
    for name, param in cams.named_parameters():
        if freeze_legacy:
            param.requires_grad = (
                name.startswith('sler_query_adapters.') or
                name == 'sler_gate_logits'
            )
        elif train_scope == 'composition':
            param.requires_grad = (
                name == 'csp.com_ctx_vectors' or
                name == 'clip_model.visual.proj_com' or
                name.startswith('clip_model.visual.tr_c.')
            )
        else:
            # Freeze the pre-trained CLIP parameters while preserving CAMS's
            # original trainable parameter set.
            param.requires_grad = any(keyword in name for keyword in keywords)

    trainable_params = [param for param in cams.parameters() if param.requires_grad]
    print(
        f'[trainable] freeze_legacy={freeze_legacy} scope={train_scope} '
        f'parameters={sum(param.numel() for param in trainable_params)}',
        flush=True,
    )
    
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    return cams, optimizer

