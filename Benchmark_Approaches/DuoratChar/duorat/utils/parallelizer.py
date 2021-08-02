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


from abc import ABC, abstractmethod

import time
import queue
import multiprocessing
import tqdm
import torch


class Parallelizer(ABC):
    """
    A parallelizer is a general purpose task used to handle the situation of
        executing functions f(x) on a series of functions and inputs f_i, x_i, where
        f is often quite large and thus difficult to transport to a given
        process, but there are typically fewer distinct values of f, allowing
        for each process to use at most one value of f.

    If it takes a constant amount of time `t_f` to ship f to a process and a constant
        amount of time `t_x` to ship x to a process then execute f(x) and return the
        result, then if we have `n` xs for a given f, then using `k` processes
        takes time `k * t_f + n/k * t_x`, which is minimized when `t_f = n * t_x / k^2`,
        or when `k = sqrt(n * t_x / t_f)`.

    If `ratio=None` is passed in, we empirically `ratio = t_f/t_x`
        with an estimate of `t_x` and `t_f`, unless there are not many
        examples, in which case 1 is used by default.
    """

    def __init__(self, max_nproc, ratio=None):
        """
        Create a Parallelizer

        Args
            nproc: the total number of processes to use
            ratio: the predicted ratio between the amount of time to set up a process
                and the amount of time to execute the function.
        """
        assert isinstance(max_nproc, int) and max_nproc >= 1
        assert ratio is None or isinstance(ratio, float) and ratio > 0
        self.max_nproc = max_nproc
        self.ratio = ratio

    @abstractmethod
    def start_worker(self, f, input_queue, output_queue):
        """
        Create a worker, and transport f to it.
        """

    @abstractmethod
    def create_queue(self):
        """
        Create a queue.Queue that interacts well with the worker threads.
        """

    def _map_pooled(self, f, n_workers, xs, pbar):
        """
        A generator that yields map(f, xs)
        """
        input = self.create_queue()
        output = self.create_queue()
        for indexed in enumerate(xs):
            input.put(indexed)

        for _ in range(n_workers):
            self.start_worker(f, input, output)

        results = [None] * len(xs)
        filled = [False] * len(xs)
        to_yield = 0
        for _ in range(len(xs)):
            index, value = output.get()
            results[index] = value
            filled[index] = True
            pbar.update()
            while to_yield < len(xs) and filled[to_yield]:
                yield results[to_yield]
                to_yield += 1

    def _compute_ratio(self, grouped_args, n_samples=5):
        total = sum(len(xs) for _, xs in grouped_args)
        if total <= n_samples * 10:
            return 1.0
        f, (x, *_) = grouped_args[0]

        start = time.time()
        input_queue = self.create_queue()
        output_queue = self.create_queue()
        for idx in range(n_samples):
            input_queue.put((idx, x))
        self.start_worker(f, input_queue, output_queue)

        output_queue.get()
        end_1 = time.time()

        for _ in range(n_samples - 1):
            output_queue.get()
        end_n = time.time()

        t_x = (end_n - end_1) / (n_samples)
        t_f = (end_1 - start) - t_x

        return max(t_f / t_x, 1e-10)  # in case t_f was negative

    def pbar(self, total):
        return tqdm.tqdm(total=total, smoothing=0, dynamic_ncols=True)

    def parallel_map(self, grouped_args):
        """
        Run the function fn on each of the args, in parallel, and yields each of
            the results.

        Args
            grouped_args: an iterable containing pairs (f, xs).
                x values are "large" in some way and will be transported as
                little as possible
        Yields
            (f(x) for f, xs in grouped_args for x in xs)
        """

        grouped_args = list(grouped_args)
        if not grouped_args:
            return
        total = sum(len(xs) for _, xs in grouped_args)

        if self.max_nproc == 1:
            pbar = self.pbar(total)
            for f, xs in grouped_args:
                for x in xs:
                    yield f(x)
                    pbar.update()
            pbar.close()
            return

        ratio = self.ratio
        if ratio is None:
            ratio = self._compute_ratio(grouped_args)
            print("Computed ratio: %.2f" % ratio)

        pbar = self.pbar(total)
        nworkers = []
        generators = []
        while grouped_args:
            f, xs = grouped_args.pop(0)
            k = min(max(1, int((len(xs) / ratio) ** 0.5)), self.max_nproc)
            while k + sum(nworkers) > self.max_nproc:
                nworkers.pop(0)
                yield from generators.pop(0)

            nworkers.append(k)
            generators.append(self._map_pooled(f, k, xs, pbar))

        for gen in generators:
            yield from gen

        pbar.close()


class CPUParallelizer(Parallelizer):
    def start_worker(self, f, input_queue, output_queue):
        worker = multiprocessing.Process(
            target=self.multi_processing_worker, args=(f, input_queue, output_queue)
        )
        worker.start()

    def create_queue(self):
        return multiprocessing.Queue()

    @staticmethod
    def multi_processing_worker(f, input_queue, output_queue):
        while True:
            try:
                index, x = input_queue.get(False)
            except queue.Empty:
                return
            output_queue.put((index, f(x)))


class CUDAParallelizer(Parallelizer):
    def __init__(self, max_nproc, ratio=None):
        self.ctx = torch.multiprocessing.get_context("spawn")
        super(CUDAParallelizer, self).__init__(max_nproc, ratio)

    def start_worker(self, f, input_queue, output_queue):
        worker = self.ctx.Process(
            target=self.multi_processing_worker, args=(f, input_queue, output_queue)
        )
        worker.start()

    def create_queue(self):
        return self.ctx.Queue()

    @staticmethod
    def multi_processing_worker(f, input_queue, output_queue):
        while True:
            try:
                index, x = input_queue.get(False)
            except queue.Empty:
                return
            while True:
                try:
                    y = f(x)
                    break
                except RuntimeError:
                    time.sleep(1)
            output_queue.put((index, y))
