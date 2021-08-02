# MIT License
#
# Copyright (c) 2019 seq2struct contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import random
import sys

import numpy as np
import torch


class RandomState:
    def __init__(self):
        self.random_mod_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_cpu_state = torch.get_rng_state()
        self.torch_gpu_states = [
            torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())
        ]

    def restore(self):
        random.setstate(self.random_mod_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_cpu_state)
        for d, state in enumerate(self.torch_gpu_states):
            torch.cuda.set_rng_state(state, d)


class RandomContext:
    """Save and restore state of PyTorch, NumPy, Python RNGs."""

    def __init__(self, seed=None):
        outside_state = RandomState()

        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            torch.manual_seed(random.randint(-sys.maxsize - 1, sys.maxsize))
        else:
            torch.manual_seed(seed)
        # torch.cuda.manual_seed_all is called by torch.manual_seed
        self.inside_state = RandomState()

        outside_state.restore()

        self._active = False

    def __enter__(self):
        if self._active:
            raise Exception("RandomContext can be active only once")

        # Save current state of RNG
        self.outside_state = RandomState()
        # Restore saved state of RNG for this context
        self.inside_state.restore()
        self._active = True

    def __exit__(self, exception_type, exception_value, traceback):
        # Save current state of RNG
        self.inside_state = RandomState()
        # Restore state of RNG saved in __enter__
        self.outside_state.restore()
        self.outside_state = None

        self._active = False
