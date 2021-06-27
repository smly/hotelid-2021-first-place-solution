import time
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()

    if logger:
        logger.info(f'[{name}] start.')
    else:
        print(f'[{name}] start.')
    yield
    if logger:
        logger.info(f'[{name}] done in {time.time() - t0:.0f} s')
    else:
        print(f'[{name}] done in {time.time() - t0:.0f} s')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
