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


import abc


class AbstractPreproc(metaclass=abc.ABCMeta):
    """Used for preprocessing data according to the model's liking.

    Some tasks normally performed here:
    - Constructing a vocabulary from the training data
    - Transforming the items in some way, such as
        - Parsing the AST
        - 
    - Loading and providing the pre-processed data to the model

    TODO:
    - Allow transforming items in a streaming fashion without loading all of them into memory first
    """

    @abc.abstractmethod
    def validate_item(self, item, section):
        """Checks whether item can be successfully preprocessed.
        
        Returns a boolean and an arbitrary object."""
        pass

    @abc.abstractmethod
    def add_item(self, item, section, validation_info):
        """Add an item to be preprocessed."""
        pass

    @abc.abstractmethod
    def clear_items(self):
        """Clear the preprocessed items"""
        pass

    @abc.abstractmethod
    def save(self):
        """Marks that all of the items have been preprocessed. Save state to disk.

        Used in preprocess.py, after reading all of the data."""
        pass

    @abc.abstractmethod
    def load(self):
        """Load state from disk."""
        pass

    @abc.abstractmethod
    def dataset(self, section):
        """Returns a torch.data.utils.Dataset instance."""
        pass
