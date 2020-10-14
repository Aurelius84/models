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


class MobileNetV1ToStatic(Benchmark):
    """
    @to_static
    """
    def __init__(self):
        """
        init
        """
        super(MobileNetV1ToStatic, self).__init__(self.__class__.__name__)
        self.model_path='mobilenetv1_d2s'

    def set_data_shape(self):
        """
        set data_shape
        """
        self.shapes = [(1, 3, 224, 224)]
        self.dtypes = [np.float32]



class MobileNetV1(MobileNetV1ToStatic):
    """
    Static graph
    """
    def __init__(self):
        """
        init
        """
        super(MobileNetV1, self).__init__()
        self.model_path='mobilenetv1'



class MobileNetV2ToStatic(MobileNetV1ToStatic):
    """
    @to_static
    """
    def __init__(self):
        """
        init
        """
        super(MobileNetV2ToStatic, self).__init__()
        self.model_path='mobilenetv2_d2s'


class MobileNetV2(MobileNetV1ToStatic):
    """
    Static graph
    """
    def __init__(self):
        """
        init
        """
        super(MobileNetV2, self).__init__()
        self.model_path='mobilenetv2'