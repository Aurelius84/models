import paddle
import paddle.fluid as fluid


from models.yolov3 import YOLOv3

def save_inference_model():

    paddle_model_dir = '/workspace/code_dev/paddle-predict/paddle/static/'

    model = YOLOv3(is_train=False)
    model.build_model()
    outputs = model.get_pred()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.save_inference_model(paddle_model_dir+"yolov3", 
                                  feeded_var_names=['image', 'im_shape'],
                                  target_vars=outputs,
                                  model_filename='model',
                                   params_filename='params',
                                  executor=exe)

def check_model():
    path = '/workspace/code_dev/paddle-predict/paddle/static/yolov3/'

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    [infer_program, feed_name, targets] = fluid.io.load_inference_model(dirname=path,
                                  executor=exe,
                                  model_filename='model',
                                  params_filename='params')
    print(feed_name, targets)
    x = np.random.normal(0.485, 0.229,
                                       [2, 3, 224, 224]).astype('float32')
    x_shape = np.array([[224, 224], [224, 224]]).astype('int32')
    outs = exe.run(infer_program, feed={feed_name[0]: x, feed_name[1]: x_shape}, fetch_list=targets, return_numpy=False)
    print(outs[0])


if __name__ == '__main__':
    save_inference_model()
