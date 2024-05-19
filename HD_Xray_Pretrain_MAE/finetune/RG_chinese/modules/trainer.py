import os
from tqdm import tqdm
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        # self.device = model.device
        
        self.model = model.to(self.device)
        # if len(device_ids) > 1:
        #     print(f'Use GPU {device_ids}...')
        #     self.model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        
        # local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # self.device = torch.device('cuda', local_rank)
        # self.model = model
            
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs

        self.mnt_mode = args.monitor_mode  # max
        self.mnt_metric = 'val_' + args.monitor_metric  # 评价指标 BLEU_4
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _test(self, save):
        raise NotImplementedError

    def test(self):
        result = self._test(save=True)
        res_str = ''
        for key, value in result.items():
            res_str += '\t{:15s}: {}\n'.format(str(key), value)
            # print('\t{:15s}: {}'.format(str(key), value))
        print(res_str)
        with open(os.path.join(self.args.save_dir, 'evaluating_indicator.txt'), mode='w', encoding='utf-8') as f:
            f.write(res_str)

    def train(self):
        # 当50个epoch没有提升自动结束训练
        not_improved_count = 0
        # for epoch in range(self.start_epoch, self.epochs + 1):
        print('start trainning!')
        epoch = self.start_epoch
        while epoch <= self.epochs:
            print('epoch:', epoch)
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            with open(os.path.join(self.args.save_dir, 'log.txt'), mode='a+', encoding='utf-8') as f:
                for key, value in log.items():
                    txt = '{}: {}'.format(str(key), value)
                    f.write(txt)
                    f.write('\n')
                    print(txt)

            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            self._save_checkpoint(epoch, save_best=best)
            epoch += 1
            print('=' * 100)
        self._print_best()
        # self._print_best_to_file()

    def _print_best_to_file(self):
        """输出最佳结果"""
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)
        
    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        # 存储断点
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        # 加载断点
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        try:
            self.mnt_best = checkpoint['monitor_best']
        except KeyError:
            self.mnt_best = inf
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.args.lr_ve
        self.optimizer.param_groups[1]['lr'] = self.args.lr_ed
        print("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            # 更新最佳结果
            self.best_recorder['val'].update(log)

        # improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
        #    self.mnt_metric_test]) or \
        #                (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
        #                    self.mnt_metric_test])
        # if improved_test:
        #    self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model

    def _train_epoch(self, epoch):
        """训练函数"""
        train_loss = 0
        # 训练标志
        self.model.train()
        par = tqdm(enumerate(self.train_dataloader))
        for batch_idx, (images_id, images, reports_ids, reports_masks) in par:
            # 将数据copy到GPU上
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

            # 重置梯度，一个batch算一次，不应该进行累计，因此batch_size越大越能够代表整个训练集，效果也越好
            self.optimizer.zero_grad()
            report_output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(report_output, reports_ids, reports_masks)
            # 得到元素张量里面的元素值 --> float
            train_loss += loss.item()
            # 反向计算梯度
            loss.backward()
            # 梯度裁剪，防止梯度爆炸，指定clip_value之后，裁剪的范围就是[-clip_value, clip_value]
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            # 更新参数
            self.optimizer.step()
            par.set_postfix(step_loss=loss.item())
        # 计算平均loss
        train_loss = train_loss / len(self.train_dataloader)
        log = {'train_loss': train_loss}

        print('eval!')
        log = self._eval(log, self.val_dataloader)
        # if epoch >= 50:
        #    print('test!')
        #    log = self._eval(log, self.test_dataloader, split='test_')

        self.lr_scheduler.step()
        return log

    def _eval(self, log, dataloader, split='val_'):
        val_gts, val_res = [], []
        self.model.eval()
        with torch.no_grad():
            par = tqdm(enumerate(dataloader))
            for batch_idx, (images_id, images, reports_ids, reports_masks) in par:
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{split + k: v for k, v in val_met.items()})
        return log

    def _test(self, save=False):
        print("Start testing!")
        self.model.eval()
        test_gts, test_res = [], []
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
        if save:
            with open(os.path.join(self.args.save_dir, 'reports.txt'), mode='w', encoding='utf-8') as f:
                total_reporter = []
                for i in range(len(test_res)):
                    reporter = "第" + str(i) + "个预测值：" + test_res[i] + "\n第" + str(i) + "个真实值：" + test_gts[i] + "\n\n"
                    total_reporter.append(reporter)
                f.write(''.join(total_reporter))
            print("Successful!")

        test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log = ({'test_' + k: v for k, v in test_met.items()})
        return log
