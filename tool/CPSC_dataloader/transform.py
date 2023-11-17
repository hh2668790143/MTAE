import random
import torch

from MlultModal.tool.CPSC_dataloader.get_noise import *


class Raw:
    def __init__(self):
        pass

    def __call__(self, data):
        print('Raw')
        return data


class Gussian:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Gussian_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Gussian'


class Rayleign:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Rayleign_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Rayleign'


class Gamma:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Gamma_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Gamma'


class Poisson:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Poisson_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Poisson'


class Exponential:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Exponential_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Exponential'


class Uniform:
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Uniform_Noisy_return_N(data, self.snr)
        return trans, noisy, 'Uniform'


class Kalman:
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)
        return data, data, 'Raw'

    def forward(self, data):
        noisy, trans, _ = Kalman_1D(data)
        return trans, noisy, 'Kalman'


class ToTensor:
    '''
    Attributes
    ----------
    basic : convert numpy to PyTorch tensor

    Methods
    -------
    forward(img=input_image)
        Convert HWC OpenCV image into CHW PyTorch Tensor
    '''

    def __init__(self, basic=False):
        self.basic = basic

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv/numpy image

        Returns
        -------
        Torch tensor
            BGR -> RGB, [0, 255] -> [0, 1]
        '''
        ret = torch.from_numpy(img).type(torch.FloatTensor)
        return ret


class Compose:
    def __init__(self, transforms, snr):
        self.transforms = transforms
        self.snr = snr

    def __call__(self, data):
        return self.forward(data)

    def fix_noisy(self, x, y):
        return x + y

    def forward(self, data):
        fix_noisy = None
        label = []
        for t in self.transforms:
            data, noisy, lb = t(data)
            label.append(lb)
            if lb == 'Raw':
                continue
            fix_noisy = noisy
            # break

        if fix_noisy is None:
            label[0] = 'Gussian'
            noisy, data, _ = Gussian_Noisy_return_N(data, self.snr)
            fix_noisy = noisy
        return data, fix_noisy, label
