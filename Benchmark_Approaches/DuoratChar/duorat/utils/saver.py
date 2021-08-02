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


"""Tools to save/restore model from checkpoints."""

import argparse
import shutil
import sys
import os
import re
import json
import logging

import torch

CHECKPOINT_PATTERN = re.compile("^model_checkpoint-(\d+)$")

logger = logging.getLogger(__name__)


class ArgsDict(dict):
    def __init__(self, **kwargs):
        super(ArgsDict, self).__init__()
        for key, value in kwargs.items():
            self[key] = value
        self.__dict__ = self


def load_checkpoint(
    model, optimizer, model_dir, load_best, map_location=None, step=None
):
    if step is not None:
        path = os.path.join(model_dir, "model_checkpoint-{:08d}".format(step))
    elif load_best:
        # Load best model
        path = os.path.join(model_dir, "model_best_checkpoint")
        if not os.path.exists(path):
            logger.warning(
                f"{path} does not exist, loading model_last_checkpoint instead ..."
            )
            path = os.path.join(model_dir, "model_last_checkpoint")
        # Backwards compatibility: load "model_checkpoint" if "model_best_checkpoint" is not available
        if not os.path.exists(path):
            logger.warning(
                f"{path} does not exist, loading model_checkpoint instead ..."
            )
            path = os.path.join(model_dir, "model_checkpoint")
    else:
        # Load last model
        path = os.path.join(model_dir, "model_last_checkpoint")
        # Backwards compatibility: load "model_checkpoint" if "model_last_checkpoint" is not available
        if not os.path.exists(path):
            logger.warning(
                f"{path} does not exist, loading model_checkpoint instead ..."
            )
            path = os.path.join(model_dir, "model_checkpoint")

    if os.path.exists(path):
        logger.info("Loading model from %s" % path)
        checkpoint = torch.load(path, map_location=map_location)
        old_state_dict = model.state_dict()
        for key in old_state_dict.keys():
            if key not in checkpoint["model"]:
                logger.warning(f"missing key {key}: using random initialization")
                checkpoint["model"][key] = old_state_dict[key]
        for key in list(checkpoint["model"].keys()):
            if key not in old_state_dict:
                logger.warning(f"unexpected key {key}")
                del checkpoint["model"][key]
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("step", 0), checkpoint.get("best_validation_metric", 0)
    return 0, 0


def load_and_map_checkpoint(model, model_dir, remap):
    path = os.path.join(model_dir, "model_checkpoint")
    print("Loading parameters %s from %s" % (remap.keys(), model_dir))
    checkpoint = torch.load(path)
    new_state_dict = model.state_dict()
    for name, value in remap.items():
        # TODO: smarter mapping.
        new_state_dict[name] = checkpoint["model"][value]
    model.load_state_dict(new_state_dict)


def save_checkpoint(model, optimizer, step, model_dir, is_best, best_validation_metric):
    """
    If is_best, save as model_best, and save model_last as a pointer to this file
    Otherwise, save as model_last
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    last_checkpoint_path = os.path.join(model_dir, "model_last_checkpoint")
    best_checkpoint_path = os.path.join(model_dir, "model_best_checkpoint")

    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_validation_metric": best_validation_metric,
        },
        best_checkpoint_path if is_best else last_checkpoint_path,
    )
    if is_best:
        if os.path.exists(last_checkpoint_path):
            os.unlink(last_checkpoint_path)
        try:
            os.symlink(os.path.basename(best_checkpoint_path), last_checkpoint_path)
        except OSError as e:
            print(e)
            shutil.copy2(best_checkpoint_path, last_checkpoint_path)


def atomic_torch_save(object_, path):
    tmp_path = path + ".tmp"
    torch.save(object_, tmp_path)
    shutil.move(tmp_path, path)


class Saver(object):
    """Class to manage save and restore for the model and optimizer."""

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    def restore(self, model_dir, map_location=None, step=None, load_best=False):
        """Restores model and optimizer from given directory.
        If load_best, loads the best model. Otherwise, loads the last model.

        Returns:
           last_step: Last training step for the model restored
           best_validation_metric: Best metric on the validation set so far
        """
        last_step, best_validation_metric = load_checkpoint(
            self._model, self._optimizer, model_dir, load_best, map_location, step
        )
        return last_step, best_validation_metric

    def save(self, model_dir, step, is_best: bool, best_validation_metric):
        """Saves model and optimizer to given directory.

        Args:
           model_dir: Model directory to save.
           step: Current training step.
           is_best: Whether the model is the best so far.
           best_validation_metric: Best metric on the validation set so far.
        """
        save_checkpoint(
            self._model,
            self._optimizer,
            step,
            model_dir,
            is_best,
            best_validation_metric,
        )

    def restore_part(self, other_model_dir, remap):
        """Restores part of the model from other directory.

        Useful to initialize part of the model with another pretrained model.

        Args:
            other_model_dir: Model directory to load from.
            remap: dict, remapping current parameters to the other model's.
        """
        load_and_map_checkpoint(self._model, other_model_dir, remap)
