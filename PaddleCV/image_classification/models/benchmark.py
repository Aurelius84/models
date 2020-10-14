import paddle
import paddle.fluid as fluid

from resnet import ResNet50, ResNet101
from mobilenet_v1 import MobileNetV1
from mobilenet_v2 import MobileNetV2

root_dir = '/workspace/all_models/'

def save_inference_model(model, model_name):
    main_prog = fluid.Program()
    start_prog = fluid.Program()
    with fluid.program_guard(main_prog, start_prog):
        img = fluid.data(shape=[None, 3, 224, 224], name='img')
        model = model()
        y = model.net(img, class_dim=1000) # align with dygraph
    
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(start_prog)

        fluid.io.save_inference_model(
            dirname=root_dir+model_name, 
            feeded_var_names=['img'], 
            target_vars=y,
            executor=exe,
            main_program=None,
            model_filename='model',
            params_filename='params')

if __name__ == '__main__':
    # save_inference_model(ResNet50, 'resnet50')
    # save_inference_model(ResNet101, 'resnet101')

    # save_inference_model(MobileNetV1, 'mobilenetv1')
    save_inference_model(MobileNetV2, 'mobilenetv2')