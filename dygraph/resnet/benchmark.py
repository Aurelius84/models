import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.jit import to_static

from train import ResNet
# from mobilenet_v1 import MobileNetV1
# from mobilenet_v2 import MobileNetV2


def save_resnet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        if '101' in model_name:
            net = model(101, class_dim=1000)
        else:
            net = model(class_dim=1000)
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])
        config = paddle.jit.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path='/workspace/all_models/'+ model_name, configs=config)

def save_mobilenet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        net = model(scale=1.0, class_dim=1000)
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])

        config = paddle.jit.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path='/workspace/all_models/'+ model_name, configs=config)
    

if __name__ == '__main__':
    save_resnet(ResNet, 'resnet50_d2s_nosoftmax')
    # save_resnet(ResNet, 'resnet101_d2s')

    # save_mobilenet(MobileNetV1, 'mobilenetv1_d2s')
    # save_mobilenet(MobileNetV2, 'mobilenetv2_d2s')