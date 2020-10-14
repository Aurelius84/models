#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from paddle import fluid
from predictor import Benchmark

core = fluid.core
logger = logging.getLogger(__name__)


class ResNet50ToStatic(Benchmark):
    """
    @to_static
    """
    def __init__(self):
        """
        init
        """
        super(ResNet50ToStatic, self).__init__(self.__class__.__name__)
        self.model_path='resnet50_d2s'

    def set_data_shape(self):
        """
        set data_shape
        """
        self.shapes = [(1, 3, 224, 224)]
        self.dtypes = [np.float32]



class ResNet50(ResNet50ToStatic):
    """
    Static graph
    """
    def __init__(self):
        """
        init
        """
        super(ResNet50, self).__init__()
        self.model_path='resnet50'


class ResNet101ToStatic(ResNet50ToStatic):
    """
    @to_static
    """
    def __init__(self):
        """
        init
        """
        super(ResNet101ToStatic, self).__init__()
        self.model_path='resnet101_d2s'


class ResNet101(ResNet50ToStatic):
    """
    Static graph
    """
    def __init__(self):
        """
        init
        """
        super(ResNet101, self).__init__()
        self.model_path='resnet101'