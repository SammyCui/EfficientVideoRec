from trainer.base import BaseTrainer
from utils.utils import args_parser, DebugArgs
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')


if __name__ == '__main__':
    args = args_parser()
    #args= DebugArgs()
    trainer = BaseTrainer(args)
    trainer.train()
    trainer.test()
    trainer.finish()


