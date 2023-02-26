import timm.models.vision_transformer
import torch
from timm.models.helpers import checkpoint_seq, resolve_pretrained_cfg, build_model_with_cfg
from torch import nn
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, checkpoint_filter_fn
from models.reducer import BaseReducer


class ReduceViT(VisionTransformer):
    def __init__(self, reducer_inner_dim, keep_ratio, **kwargs):
        super().__init__(**kwargs)
        self.reducer = BaseReducer(patch_size=kwargs.get('patch_size'),
                                   in_chans=kwargs.get('in_chans', 3),
                                   dim=reducer_inner_dim,
                                   keep_ratio=keep_ratio)

    def forward_features(self, x):
        keep_ind = self.reducer(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = x[torch.arange(x.shape[0]).unsqueeze(1), keep_ind]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

def reducer_vit_tiny_patch16_224(args):
    variant = 'vit_tiny_patch16_224'
    assert args.patch_size == 16, 'patch_size has to be 16 for vit tiny'
    # model = ReduceViT(reducer_inner_dim=args.reducer_inner_dim,
    #                     keep_ratio=args.keep_ratio,
    #                     patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=args.num_classes, in_chans=3)

    pretrained_cfg = resolve_pretrained_cfg(variant)
    model = build_model_with_cfg(ReduceViT, variant,
                                 pretrained=args.pretrained,
                                 pretrained_cfg=pretrained_cfg,
                                 pretrained_custom_load='npz' in pretrained_cfg['url'],
                                 reducer_inner_dim=args.reducer_inner_dim,
                                 keep_ratio=args.keep_ratio,
                                 patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=args.num_classes,
                                 in_chans=3
                                 )
    if args.pretrained:
        train_params = ['reducer', 'head']
        for name, param in model.named_parameters():
            param.requires_grad = True if any(train_param in name for train_param in train_params) else False

    return model


def vit_tiny_patch16_224(args):
    model = timm.models.vision_transformer.vit_tiny_patch16_224(pretrained=args.pretrained, num_classes=args.num_classes)

    if args.pretrained:
        train_params = ['reducer', 'head']
        for name, param in model.named_parameters():
            param.requires_grad = True if any(train_param in name for train_param in train_params) else False

    return model


if __name__ == '__main__':
    model = timm.models.vision_transformer.vit_tiny_patch16_224(pretrained=True, num_classes=10)
    print(model)