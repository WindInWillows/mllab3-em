# -*- coding: utf-8 -*-

import numpy as np


class guass():
    def __init__(self, miu=None, sigma=None, dim=2):
        if miu is None :
            self.miu = np.zeros(dim)
        else:
            self.miu = miu
        if sigma is None:
            self.sigma = np.zeros(dim, dim)
        else:
            self.sigma = sigma
        self.dim = dim


