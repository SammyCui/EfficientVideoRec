import abc
import json
import torch
import os
import torch.nn.functional as F
from utils.logger import Logger
from utils.meters import AverageMeter, StatsMeter
from utils.metric import accuracy
from timeit import default_timer as timer

from utils.helpers import get_model_optimizer, get_dataloaders


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.logger = Logger(args)
        self.args = args
        self.train_step = 0
        self.train_epoch = args.start_epoch
        self.start_epoch = args.start_epoch
        self.max_epoch = args.max_epoch
        self.device = args.device

        self.train_time, self.forward_tm, self.backward_tm, self.optimize_tm = (AverageMeter(),) * 4

        self.result_log = {'max_val_acc': 0,
                           'max_val_acc_epoch': 0}

        self.model, self.optimizer, self.lr_scheduler = get_model_optimizer(args)
        self.model.to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(args)
        self.best_model_params = None
        print(json.dumps(vars(args), indent=2))

    def train(self):
        print('==> Training Start')
        epoch_t0 = timer()
        for epoch in range(self.start_epoch, self.max_epoch+1):

            self.model.train()

            train_loss, train_acc = AverageMeter(), AverageMeter()

            for batch in self.train_dataloader:

                self.train_step += 1

                if self.args.per_size:
                    data, bb, labels = batch
                    data, bb, labels = data.to(self.device), bb.to(self.device), labels.to(self.device)
                    data = (data, bb)
                else:
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                forward_t0 = timer()
                outputs = self.model(*data)
                forward_t1 = timer()

                loss = F.cross_entropy(outputs, labels)
                acc  = accuracy(outputs, labels)[0]

                self.optimizer.zero_grad()
                backward_t0 = timer()
                loss.backward()
                backward_t1 = timer()

                optimizer_t0 = timer()
                self.optimizer.step()
                optimizer_t1 = timer()

                train_loss.update(loss.item())
                train_acc.update(acc[0].item())
                self.backward_tm.update(backward_t1-backward_t0)
                self.forward_tm.update(forward_t1-forward_t0)
                self.optimize_tm.update(optimizer_t1-optimizer_t0)

            if self.lr_scheduler:
                self.lr_scheduler.step()
            val_acc, val_loss = self.validate()

            epoch_t1 = timer()
            self.train_time.update(epoch_t1-epoch_t0)
            epoch_t0 = epoch_t1

            self.logging(train_loss=train_loss.avg, train_acc=train_acc.avg,
                         val_loss=val_loss, val_acc=val_acc)
            self.train_epoch += 1

    def _validate(self):
        self.model.eval()
        val_loss, val_acc = StatsMeter(), StatsMeter()
        with torch.no_grad():
            for batch in self.val_dataloader:
                if self.args.per_size:
                    data, bb, labels = batch
                    data, bb, labels = data.to(self.device), bb.to(self.device), labels.to(self.device)
                    data = (data, bb)
                else:
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(*data)

                loss = F.cross_entropy(outputs, labels)
                acc = accuracy(outputs, labels)[0]

                val_loss.update(loss.item())
                val_acc.update(acc[0].item())


        return val_acc.avg, val_loss.avg

    def validate(self):
        if self.train_epoch % self.args.val_interval == 0:
            val_acc, val_loss = self._validate()

            if val_acc >= self.result_log['max_val_acc']:
                self.result_log['max_val_acc'] = val_acc
                self.result_log['max_val_acc_epoch'] = self.train_epoch
                self.best_model_params = self.model.state_dict()
                if self.args.save:
                    self.save_model('checkpoint')

            return val_acc, val_loss

    def save_model(self, name):
        assert self.model is not None, "No models to be saved."
        checkpoint = {'models': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if self.lr_scheduler:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.args.result_dir, name + '.pt'))

    def logging(self,
                train_loss,
                train_acc,
                val_loss,
                val_acc):
        assert self.optimizer is not None, "Has not initialize optimizer yet."

        if self.train_epoch % self.args.val_interval == 0:
            print('epoch {}/{}, **Train** loss={:.4f} acc={:.4f} | ' \
                  '**Val** loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.max_epoch,
                          train_loss, train_acc,
                          val_loss, val_acc,
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_loss', train_loss, self.train_epoch)
            self.logger.add_scalar('train_acc', train_acc, self.train_epoch)
            self.logger.add_scalar('val_loss', val_loss, self.train_epoch)
            self.logger.add_scalar('val_acc', val_acc, self.train_epoch)

    def finish(self):
        self.logger.save_logger()
        print("==>", 'Training Statistics')

        for k, v in self.result_log.items():
            print(k, ': ', '{:.3f}'.format(v))

        print(
            'forward_timer  (avg): {:.2f} sec  \n' \
            'backward_timer (avg): {:.2f} sec, \n' \
            'optim_timer (avg): {:.2f} sec \n' \
            'epoch_timer (avg): {:.5f} hrs \n' \
            'total time to converge: {:.2f} hrs' \
                .format(
                    self.forward_tm.avg, self.backward_tm.avg,
                    self.optimize_tm.avg, self.train_time.avg / 3600,
                    self.train_time.sum / 3600
                )
        )
        with open(os.path.join(self.args.result_dir, 'results.txt'), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f}\n'.format(
                self.result_log['max_val_acc_epoch'],
                self.result_log['max_val_acc']))
            f.write('Test acc={:.4f}\n'.format(
                self.result_log['test_acc']))
            f.write('Total time to converge: {:.3f} hrs, per epoch: {:.5f} hrs'
                    .format(self.train_time.sum / 3600, self.train_time.avg / 3600))

        self.logger.close()

    def test(self):
        print('==> Testing start')
        self.model.load_state_dict(self.best_model_params)
        self.model.eval()

        test_loss, test_acc = StatsMeter(), StatsMeter()
        with torch.no_grad():
            for batch in self.val_dataloader:
                if self.args.per_size:
                    data, bb, labels = batch
                    data, bb, labels = data.to(self.device), bb.to(self.device), labels.to(self.device)
                    data = (data, bb)
                else:
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(*data)

                loss = F.cross_entropy(outputs, labels)
                acc = accuracy(outputs, labels)[0]

                test_loss.update(loss.item())
                test_acc.update(acc[0].item())


        self.result_log['test_acc'] = test_acc.avg
        self.result_log['test_loss'] = test_loss.avg

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )


