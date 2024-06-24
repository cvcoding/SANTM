import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # noqa
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
from train_agent import TrainAgent
from test_agent import TestAgent
from prepare_data import load_data

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # 打印出所有GPU的详细信息
#     for gpu in gpus:
#         print(gpu)
#     # 使用GPU
#     tf.config.experimental.set_memory_growth(gpus[0], True)
# else:
#     print("No GPU available")
#     a=1

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 设置GPU内存增长
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)


def parse_args():
    '''
        parsing and configuration
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str,
                        default="standard1",
                        help="[%(default)s] A string to describe this model")
    parser.add_argument("--data", type=str,
                        default='mnist',
                        choices=['pmnist', 'mnist', 'add', 'copy'],
                        help="[%(default)s] Path to the dataset.")
    parser.add_argument("--layers", type=str,
                        default="32, 32",  # actually 3 layers are used 32-32-32
                        # allocate too much computation resource.
                        help="[%(default)s] A comma-separated list"
                        " of the layer sizes")
    parser.add_argument("--batch_size", type=int,
                        default=50,
                        help="[%(default)s] The batch size to train with")
    parser.add_argument("--keep_prob", type=float,
                        default=0.9,
                        help='[%(default)s] The keep probability to use'
                        ' for training')
    parser.add_argument('--max_grad_norm', type=float,
                        default=5.0,
                        help='[%(default)s] The maximum grad norm to clip by')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01,  # 0.01
                        help='[%(default)s] The learning rate to train with')
    parser.add_argument('--optimizer', type=str,
                        default='adam',
                        choices=['momentum', 'rms', 'adam'],
                        help='[%(default)s] The optimizer to train with')
    parser.add_argument("--epochs", type=int,
                        default=100,
                        help="[%(default)s] The number of epochs to train for")
    parser.add_argument("--test", action='store_true',
                        help="[False] If True, the model "
                        "is only tested and not trained.")
    parser.add_argument("--logdir", type=str,
                        default="log",
                        help="[%(default)s] The directory to write"
                        " tensoboard logs to")
    parser.add_argument("--gpu", type=str,
                        default='0',
                        help="[%(default)s] The specific GPU to train on.")
    parser.add_argument('--wd', type=float,
                        default=0.0,
                        help='[%(default)s] weight decay importance')
    parser.add_argument('--results_file', type=str,
                        default='None',
                        help='[%(default)s] The file to append results to. '
                        ' If set, nothing else will be logged or saved.')
    parser.add_argument('--chrono', action='store_true',
                        help='[False] If set, chrono-initialization is used.')
    parser.add_argument('--log_test', action='store_true',
                        help='[False] Log test data metrics on TB.')
    parser.add_argument('--cell', type=str,
                        default='Reslstm',
                        choices=['Reslstm','lstm','rnn'],
                        help='[%(default)s] The type of cell to use.')
    parser.add_argument("--T", type=int,
                        default=120,
                        help="[%(default)s] Sequence length for add/copy.")
    parser.add_argument("--log_every", type=int,
                        default=200000,
                        help="[%(default)s] How often to log highres loss.")

    return parser.parse_args()


def test_wrapper(test_agent, args):
    data_list = load_data(args.data)
    x_test = data_list[4]
    y_test = data_list[5]
    test_agent.test(x_test, y_test, 'models/'+args.name+'/')


def main(args):
    if args.test:
        # Get the config
        with open(os.path.join('models',args.name,'config.json')) as fp:
            config_dict = json.load(fp)
        args_dict = vars(args)
        args_dict.update(config_dict)

        test_agent = TestAgent(args)
        test_wrapper(test_agent, args)
    else:
        with tf.device('/GPU:0'):
            agent = TrainAgent(args)
            test_agent = TestAgent(args)
        try:
            agent.train(args.data, args.max_grad_norm, args.wd,
                        test_agent, args=args)
        except KeyboardInterrupt:
            test_wrapper(test_agent, args)


if __name__ == "__main__":
    main(args=parse_args())
