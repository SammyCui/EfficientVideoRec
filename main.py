from trainer.base import BaseTrainer
from trainer.helpers import args_parser, DebugArgs

# libgcc_s = ctypes.CDLL('libgcc_s.so.1')


if __name__ == '__main__':
    # args = args_parser()
    args= DebugArgs(model='reducer_vit_tiny_patch16_224')
    trainer = BaseTrainer(args)
    if args.train:
        trainer.train()
    trainer.test(trainer.best_model_params)
    trainer.finish()


