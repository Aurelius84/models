import collections
import paddle.fluid as fluid

from resnet import ResNet50, ResNet50ToStatic, ResNet101, ResNet101ToStatic
from mobilenet import MobileNetV1, MobileNetV1ToStatic, MobileNetV2, MobileNetV2ToStatic

root_dir = '/workspace/code_dev/paddle-predict/paddle/'


def load_model(model_path):
    exe = fluid.Executor(fluid.CPUPlace())
    [infer_prog, feed_name, fetch_vars] = fluid.io.load_inference_model(root_dir+model_path, executor=exe, model_filename='model', params_filename='params')
    vars = infer_prog.global_block().vars
    # print(infer_prog)
    info = {}
    for var_name in vars:
        try:
            info[var_name] = vars[var_name].shape
        except:
            pass
    return infer_prog, feed_name, fetch_vars, info

def calculate_op_nums(program):
    ops = []
    for block_id in range(len(program.blocks)):
        block = program.block(block_id)
        cur_block_ops = [op.type for op in block.ops]
        ops.extend(cur_block_ops)
    
    op_nums = collections.Counter(ops)
    return op_nums


def print_ops_nums(model_path):
    infer_prog,_,_, info = load_model(model_path)
    op_nums = calculate_op_nums(infer_prog)
    print(sorted(op_nums.items()))
    return info

if __name__ == '__main__':
    # print_ops_nums('dy2stat/resnet50')
    # print_ops_nums('static/resnet50')

    print_ops_nums('static/seq2seq')
    print_ops_nums('dy2stat/seq2seq_1')
    print_ops_nums('dy2stat/seq2seq_1_yes')
    

    # print(info1)
    # print(info2)

    # print_ops_nums('dy2stat/mobilenetv1')
    # print_ops_nums('static/mobilenetv1')

    # info1 = print_ops_nums('dy2stat/yolov3')
    # info1 = print_ops_nums('static/yolov3')
    # info2 = print_ops_nums('static/yolov3_darknet')
    # for var_name, shape in info2.items():
    #     # print(var_name, shape)
    #     if var_name not in info2:
    #         print('error 1: {} {} not in info2'.format(var_name, shape))
    #     else:
    #         if shape != info2[var_name]:
    #             print("error: name: {}  1: {}, 2: {}".format(var_name, shape, info2[var_name]))
    #         else:
    #             print("name: {}  1: {},".format(var_name, shape, info2[var_name]))
    
    
    