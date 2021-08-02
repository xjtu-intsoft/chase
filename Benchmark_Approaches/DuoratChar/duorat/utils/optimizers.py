import attr
import math
import torch

from duorat.utils import registry

registry.register("optimizer", "adadelta")(torch.optim.Adadelta)
registry.register("optimizer", "adam")(torch.optim.Adam)
registry.register("optimizer", "sgd")(torch.optim.SGD)


@registry.register("lr_scheduler", "warmup_polynomial")
@attr.s
class WarmupPolynomialLRScheduler:
    optimizer = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()
    power = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        elif current_step < (self.num_warmup_steps + self.decay_steps):
            new_lr = (self.start_lr - self.end_lr) * (
                1 - (current_step - self.num_warmup_steps) / self.decay_steps
            ) ** self.power + self.end_lr
        else:
            new_lr = self.end_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


@registry.register("lr_scheduler", "bert_warmup_polynomial")
@attr.s
class BertWarmupPolynomialLRScheduler(WarmupPolynomialLRScheduler):
    bert_factor = attr.ib()

    def update_lr(self, current_step):
        super(BertWarmupPolynomialLRScheduler, self).update_lr(current_step)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bert":
                param_group["lr"] /= self.bert_factor


@registry.register("lr_scheduler", "warmup_cosine")
@attr.s
class WarmupCosineLRScheduler:
    optimizer = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (self.start_lr - self.end_lr) * 0.5 * (
                1
                + math.cos(
                    math.pi * (current_step - self.num_warmup_steps) / self.decay_steps
                )
            ) + self.end_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


@registry.register("lr_scheduler", "noop")
class NoOpLRScheduler:
    def __init__(self, optimizer):
        pass

    def update_lr(self, current_step):
        pass
