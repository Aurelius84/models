
import paddle.fluid as fluid
from lm_model import lm_model

import numpy as np


def main():
    # define train program
    
    # with fluid.program_guard(main_program, startup_program):
    #     with fluid.unique_name.guard():
    res_vars = lm_model(
        hidden_size=200,
        vocab_size=10000,
        num_layers=2,
        num_steps=20,
        init_scale=0.1,
        dropout=None,
        rnn_model='static')
    proj, hid, last, feed_list = res_vars
    
    paddle_model_dir = '/workspace/code_dev/paddle-predict/paddle/static/'

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(startup_program)
    fluid.io.save_inference_model(dirname=paddle_model_dir+'ptb_lm',
                                feeded_var_names = feed_list,
                                target_vars = [proj, hid, last],
                                executor=exe,
                                main_program=main_program,
                                model_filename='model',
                                params_filename='params')

    x = np.arange(80).reshape(4, 20).astype('int64')
    init_hidden = np.zeros((2, 4, 200), dtype='float32')
    init_cell = np.zeros((2, 4, 200), dtype='float32')
    print(main_program)

    out = exe.run(main_program, feed={'x':x, 'init_hidden':init_hidden, 'init_cell': init_cell}, fetch_list=[proj])
    print(out[0].shape)


main()