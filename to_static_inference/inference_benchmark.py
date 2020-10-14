#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from resnet import ResNet50, ResNet50ToStatic, ResNet101, ResNet101ToStatic
from mobilenet import MobileNetV1, MobileNetV1ToStatic, MobileNetV2, MobileNetV2ToStatic


def main():
    """
    main
    """
    root_dir = '/workspace/all_models/'
    
    models_dict = {
        "resnet50": [ResNet50, ResNet50ToStatic],
        "resnet101": [ResNet101, ResNet101ToStatic],
        "mobilenetv1": [MobileNetV1, MobileNetV1ToStatic],
        "mobilenetv2": [MobileNetV2, MobileNetV2ToStatic]
    }
    args = parse_args()
    models = models_dict.get(args.model)
    for model in models:
        model = model()  # instance
        model.set_config(
            use_gpu=args.device == 'gpu',
            model_dir=root_dir + model.model_path,
            model_filename=args.model_filename,
            params_filename=args.params_filename,
            use_tensorrt=args.use_tensorrt,
            use_anakin = args.use_anakin,
            model_precision = args.model_precision)
        model.set_data_shape()
        warmup = args.warmup
        repeat = args.repeat
        for i in range(3):
            outputs, avg_time = model.run(warmup, repeat)
            print('{} cost time : {} ms'.format(model.model_path, avg_time))


def parse_args(prog=None):
    """
    parse_args
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_dir", type=str, help="model dir")
    parser.add_argument("--model_filename", default='model', type=str, help="model filename")
    parser.add_argument("--params_filename", default='params', type=str, help="params filename")
    parser.add_argument("--device", choices=["cpu", "gpu"])
    parser.add_argument("--use_tensorrt", action='store_true', help='If set, run the model in tensorrt')
    parser.add_argument("--use_anakin", action='store_true', help='If set, run the model use anakin')
    parser.add_argument("--model_precision", type=str, default="float", choices=["float", "int8"], help='If set, run the model use anakin')
    parser.add_argument("--filename", type=str, default="data/image.jpg", help="data path")
    parser.add_argument("--warmup", type=int, default=10, help="warmup")
    parser.add_argument("--repeat", type=int, default=1000, help="repeat times")
    return parser.parse_args()


if __name__ == "__main__":
    main()
