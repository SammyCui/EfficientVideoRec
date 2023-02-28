import timm.models.vision_transformer
import torch
from timm.models.helpers import checkpoint_seq, resolve_pretrained_cfg, build_model_with_cfg
from torch import nn
from timm.models.vision_transformer import VisionTransformer
from models.reducer import BaseReducer, RandomReducer, ConvReducer


class ReduceViT(VisionTransformer):
    def __init__(self, reducer, reducer_inner_dim, keep_ratio, reducer_depth, image_size=224, **kwargs):
        super().__init__(**kwargs)
        self.reducer = eval(reducer)(patch_size=kwargs.get('patch_size'),
                                   in_chans=kwargs.get('in_chans', 3),
                                   dim=reducer_inner_dim,
                                   keep_ratio=keep_ratio,
                                   reducer_depth=reducer_depth,
                                   image_size=image_size)

    def forward_features(self, x):
        keep_ind = self.reducer(x)
        # offset indices by 1 for cls token
        keep_ind = torch.cat((torch.zeros((x.shape[0],1), device=x.device), keep_ind + 1), dim=-1)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = x[torch.arange(x.shape[0]).unsqueeze(1), keep_ind.long()]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

variant_cfg = {
    'vit_tiny_patch16_224': {'patch_size':16, 'embed_dim':192, 'depth':12, 'num_heads':3, 'in_chans':3},
    'vit_small_patch32_224': {'patch_size':32, 'embed_dim':384, 'depth':12, 'num_heads':6, 'in_chans':3},
    'vit_small_patch16_224': {'patch_size':16, 'embed_dim':384, 'depth':12, 'num_heads':6, 'in_chans':3},
    'vit_base_patch32_224': {'patch_size':32, 'embed_dim':768, 'depth':12, 'num_heads':12, 'in_chans':3},
    'vit_base_patch16_224': {'patch_size':16, 'embed_dim':768, 'depth':12, 'num_heads':12, 'in_chans':3}

}

def reducer_vit(args):
    variant = args.model_variant
    pretrained_cfg = resolve_pretrained_cfg(variant)
    model = build_model_with_cfg(ReduceViT, variant,
                                 pretrained=args.pretrained,
                                 pretrained_cfg=pretrained_cfg,
                                 pretrained_custom_load='npz' in pretrained_cfg['url'],
                                 reducer=args.reducer,
                                 reducer_inner_dim=args.reducer_inner_dim,
                                 keep_ratio=args.keep_ratio,
                                 reducer_depth=args.reducer_depth,
                                 image_size=args.image_size,
                                 num_classes=args.num_classes,
                                 **variant_cfg[args.model_variant]
                                 )
    if args.pretrained:
        train_params = ['reducer', 'head']
        for name, param in model.named_parameters():
            param.requires_grad = True if any(train_param in name for train_param in train_params) else False

    return model


def benchmark_vit(args):
    model = eval(f'timm.models.vision_transformer.{args.model_variant}')(pretrained=args.pretrained, num_classes=args.num_classes)

    if args.pretrained:
        train_params = ['reducer', 'head']
        for name, param in model.named_parameters():
            param.requires_grad = True if any(train_param in name for train_param in train_params) else False

    return model


if __name__ == '__main__':
    model = timm.models.vision_transformer.vit_tiny_patch16_224(pretrained=True, num_classes=10)
    print(model)