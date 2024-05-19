import os
import pickle
import datetime
import time
# from contextlib import contextmanger
import torch
from torch.autograd import Variable
import random
import numpy as np


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H_%M_%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


def is_iterable(obj):
    return hasattr(obj, '__len__')


def to_scalar(vt):
    """
    preprocess a 1-length pytorch Variable or Tensor to scalar
    """
    # if isinstance(vt, Variable):
    #     return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        if vt.dim() == 0:
            return vt.detach().cpu().numpy().flatten().item()
        else:
            return vt.detach().cpu().numpy()
    elif isinstance(vt, np.ndarray):
        return vt
    else:
        raise TypeError('Input should be a ndarray or tensor')


# makes the random numbers predictable
def set_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.enabled = True
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def select_gpus(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus != '':
        print("Using gpu's: {}".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        print('Using default gpu.')

def may_mkdirs(dir_name):
    # if not os.cam_path.exists(os.cam_path.dirname(os.cam_path.abspath(fname))):
    #     os.makedirs(os.cam_path.dirname(os.cam_path.abspath(fname)))
    if not os.path.exists(os.path.abspath(dir_name)):
        os.makedirs(os.path.abspath(dir_name))


class AverageMeter(object):
    """ 
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-20)


class RunningAverageMeter(object):
    """
    Computes and stores the running average and current value
    """

    def __init__(self, hist=0.99):
        self.val = None
        self.avg = None
        self.hist = hist

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val


class RecentAverageMeter(object):
    """
    Stores and computes the average of recent values
    """

    def __init__(self, hist_size=100):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, value):
        self.val = value
        self.fifo.append(value)
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)


class ReDirectSTD(object):
    """
    overwrites the sys.stdout or sys.stderr
    Args:
      fpath: file cam_path
      console: one of ['stdout', 'stderr']
      immediately_visiable: False
    Usage example:
      ReDirectSTD('stdout.txt', 'stdout', False)
      ReDirectSTD('stderr.txt', 'stderr', False)
    """

    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        import sys
        import os
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visiable = immediately_visiable

        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, **args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            if not os.path.exists(os.path.dirname(os.path.abspath(self.file))):
                os.mkdir(os.path.dirname(os.path.abspath(self.file)))

            if self.immediately_visiable:
                # open for writing, appending to the end of the file if it exists
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')

                # print("self.f is not none")
                # first time self.f is None, second is not None
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1


def set_devices(sys_device_ids):
    """
    Args:
        sys_device_ids: a tuple; which GPUs to use
          e.g.  sys_device_ids = (), only use cpu
                sys_device_ids = (3,), use the 4-th gpu
                sys_device_ids = (0, 1, 2, 3,), use the first 4 gpus
                sys_device_ids = (0, 2, 4,), use the 1, 3 and 5 gpus
    """
    import os
    visiable_devices = ''
    for i in sys_device_ids:
        visiable_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visiable_devices
    # Return wrappers
    # Models and user defined Variables/Tensors would be transferred to 
    # the first device
    device_id = 0 if len(sys_device_ids) > 0 else -1


def transfer_optims(optims, device_id=-1):
    for optim in optims:
        if isinstance(optim, torch.optim.Optimizer):
            transfer_optim_state(optim.state, device_id=device_id)


def transfer_optim_state(state, device_id=-1):
    """
    Transfer an optimizer.state to cpu or specified gpu, which means
    transferring tensors of the optimizer.state to specified device.
    The modification is in place for the state.
    Args:
        state: An torch.optim.Optimizer.state
        device_id: gpu id, or -1 which means transferring to cpu
    """
    for key, val in state.items():
        if isinstance(val, dict):
            transfer_optim_state(val, device_id=device_id)
        elif isinstance(val, Variable):
            raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
        elif isinstance(val, torch.nn.Parameter):
            raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
        else:
            try:
                if device_id == -1:
                    state[key] = val.cpu()
                else:
                    state[key] = val.cuda(device=device_id)
            except:
                pass


def load_state_dict(model, src_state_dict):
    """
    copy parameter from src_state_dict to models
    Arguments:
        model: A torch.nn.Module object
        src_state_dict: a dict containing parameters and persistent buffers
    """
    from torch.nn import Parameter
    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception as msg:
            print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        for n in src_missing:
            print('\t', n)

    dest_missint = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missint):
        print("Keys not found in destination state_dict: ")
        for n in dest_missint:
            print('\t', n)


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
    """
    load state_dict of module & optimizer from file
    Args:
        modules_optims: A two-element list which contains module and optimizer
        ckpt_file: the check point file 
        load_to_cpu: Boolean, whether to preprocess tensors in models & optimizer to cpu type
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        m.load_state_dict(sd)
    if verbose:
        print("Resume from ckpt {}, \nepoch: {}, scores: {}".format(
            ckpt_file, ckpt['ep'], ckpt['scores']))
    return ckpt['ep'], ckpt['scores']


def save_ckpt(model, ckpt_files, epoch, metric):
    """
    Note:
        torch.save() reserves device type and id of tensors to save.
        So when loading ckpt, you have to inform torch.load() to load these tensors
        to cpu or your desired gpu, if you change devices.
    """

    if not os.path.exists(os.path.dirname(os.path.abspath(ckpt_files))):
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_files)))

    save_dict = {'state_dicts': model.state_dict(),
                 'epoch': f'{time_str()} in epoch {epoch}',
                 'metric': metric}
    torch.save(save_dict, ckpt_files)


def adjust_lr_staircase(param_groups, base_lrs, ep, decay_at_epochs, factor):
    """ Multiplied by a factor at the beging of specified epochs. Different
        params groups specify thier own base learning rates.
    Args:
        param_groups: a list of params
        base_lrs: starting learning rate, len(base_lrs) = len(params_groups)
        ep: current epoch, ep >= 1
        decay_at_epochs: a list or tuple; learning rates are multiplied by a factor 
          at the begining of these epochs
        factor: a number in range (0, 1)
    Example:
        base_lrs = [0.1, 0.01]
        decay_at_epochs = [51, 101]
        factor = 0.1
    Note:
        It is meant to be called at the begining of an epoch
    """
    assert len(base_lrs) == len(param_groups), \
        'You should specify base lr for each param group.'
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)
    for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
        g['lr'] = base_lr * factor ** (ind + 1)
        print('=====> Param group {}: lr adjusted to {:.10f}'.format(i, g['lr']).rstrip('0'))


def may_set_mode(maybe_modules, mode):
    """
    maybe_modules, an object or a list of objects.
    """
    assert mode in ['train', 'eval']
    if not is_iterable(maybe_modules):
        maybe_modules = [maybe_modules]
    for m in maybe_modules:
        if isinstance(m, torch.nn.Module):
            if mode == 'train':
                m.train()
            else:
                m.eval()


def get_topk(matrix, k):
    """
    retain topk elements of a matrix and set others 0
    Args:
        matrix (object): np.array 2d
    """
    vector = matrix.reshape(matrix.shape[0] * matrix.shape[1])
    vector.sort()
    vector.get

    return matrix


class Timer:

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
