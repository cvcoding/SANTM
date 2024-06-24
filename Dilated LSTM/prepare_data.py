from __future__ import print_function
import numpy as np
from aux_code.ops import randomly_split_data
from aux_code.ops import one_hot_sequence
from six.moves import cPickle as pickle
import numpy as np
import os
import tensorflow as tf
# # from scipy.misc import imread
# from imageio import imread
import platform
from sklearn import preprocessing


def load_data(dataset_name, seq_len=200):
    '''
    Returns:
    x - a n_samples long list containing arrays of shape (sequence_length,
                                                          n_features)
    y - an array of the labels with shape (n_samples, n_classes)
    '''
    print("Loading " + dataset_name + " dataset ...")

    if dataset_name == 'test':
        n_data_points = 5000
        sequence_length = 100
        n_features = 1
        x = list(np.random.rand(n_data_points, sequence_length, n_features))
        n_classes = 4
        y = np.random.randint(low=0, high=n_classes, size=n_data_points)

    if dataset_name == 'mnist':
        return get_mnist(permute=False)

    if dataset_name == 'pmnist':
        return get_mnist(permute=True)
    if dataset_name == 'fashion':
        return get_fashion(permute=False)
    if dataset_name == 'cifar10':
        return get_cifar()

    if dataset_name == 'add':
        x, y = get_add(n_data=50000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=150000, seq_len=seq_len)

    train_idx, valid_idx, test_idx = randomly_split_data(
        y, test_frac=0.2, valid_frac=0.1)

    x_train = [x[i] for i in train_idx]
    y_train = y[train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_valid = y[valid_idx]
    x_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_add(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 2))
    x[:,:,0] = np.random.uniform(low=-0.5, high=0.5, size=(n_data, seq_len))
    inds = np.random.randint(seq_len/10, size=(n_data, 2))
    inds[:,1] += (seq_len*4)//5
    for i in range(n_data):
        x[i,inds[i,0],1] = 1.0
        x[i,inds[i,1],1] = 1.0

    y = (x[:,:,0] * x[:,:,1]).sum(axis=1)
    y = np.reshape(y, (n_data, 1))
    return x, y


def get_copy(n_data, seq_len):
    x = np.zeros((n_data, seq_len+1+2*10))
    info = np.random.randint(1, high=9, size=(n_data, 10))

    x[:,:10] = info
    x[:,seq_len+10] = 9*np.ones(n_data)

    y = np.zeros_like(x)
    y[:,-10:] = info

    x = one_hot_sequence(x)
    y = one_hot_sequence(y)

    n_train, n_valid, n_test = [100000, 10000, 40000]
    x_train = list(x[:n_train])
    y_train = y[:n_train]
    x_valid = list(x[n_train:n_train+n_valid])
    y_valid = y[n_train:n_train+n_valid]
    x_test = list(x[-n_test:])
    y_test = y[-n_test:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch1(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

        index1 = range(0, 32, 2)
        X= X[:, index1, :, :]
        X_new = X[:, :, index1, :]

        # index2 = range(0, 16, 3)
        # X_new = X_new[:, index2, :, :]
        # X_new = X_new[:, :, index2, :]

        # X = X_new.reshape(10000, 256, 3) / 255
        # X = X_new.reshape(10000, 768, 1)/255

        # X = X_new.reshape(10000, 108, 1) / 255
        X = X_new.reshape(10000, 1024, 3) / 255
        mean = np.mean(X, axis=1)
        mean = np.expand_dims(mean, axis=1)
        X = X-mean
        Y = np.array(Y)
        Y = Y.reshape(10000, 1)
        one_hot = preprocessing.OneHotEncoder(sparse=False)
        Y = one_hot.fit_transform(Y)
        return X, Y


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = np.resize(X, (10000, 3, 16, 16))
        # X = X.reshape(10000, 3072, 1)/255
        X = X.reshape(10000, 256, 3) / 255
        mean = np.mean(X, axis=1)
        mean = np.expand_dims(mean, axis=1)
        X = X-mean
        Y = np.array(Y)
        Y = Y.reshape(10000, 1)
        one_hot = preprocessing.OneHotEncoder(sparse=False)
        Y = one_hot.fit_transform(Y)
        return X, Y


# def load_CIFAR_batch_test(filename, mean):
#     """ load single batch of cifar """
#     with open(filename, 'rb') as f:
#         datadict = load_pickle(f)  # dict类型
#         X = datadict['data']  # X, ndarray, 像素值
#         Y = datadict['labels']  # Y, list, 标签, 分类
#         X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
#         X = X.reshape(10000, 3072, 1)/255
#         X = X-mean
#         Y = np.array(Y)
#         Y = Y.reshape(10000, 1)
#         one_hot = preprocessing.OneHotEncoder(sparse=False)
#         Y = one_hot.fit_transform(Y)
#         return X, Y


def get_cifar():
    """ load all of cifar """
    ROOT = "CIFAR"
    xs = []  # list
    ys = []
    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y
    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_mnist2(permute=False):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST", one_hot=True)

    if permute:
        perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    else:
        perm_mask = np.arange(784)

    x_train = list(np.expand_dims(mnist.train.images[:, perm_mask], -1))
    y_train = mnist.train.labels
    x_valid = list(np.expand_dims(mnist.validation.images[:, perm_mask], -1))
    y_valid = mnist.validation.labels
    x_test = list(np.expand_dims(mnist.test.images[:, perm_mask], -1))
    y_test = mnist.test.labels

    print("Train:Validation:Testing - %d:%d:%d" % (len(y_train), len(y_valid),
                                                   len(y_test)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# def get_fashion(permute=False):
#     mnist = tf.keras.datasets.fashion_mnist
#     (training_images, y_train), (test_images, y_test) = mnist.load_data()
#     x_train = training_images / 255.0
#     x_test = test_images / 255.0
#     # x_train = np.resize(x_train, (60000, 14, 14))
#     x_train = np.reshape(x_train, (-1, 784, 1))
#
#     # x_test = np.resize(x_test, (10000, 14, 14))
#     x_test = np.reshape(x_test, (-1, 784, 1))
#
#     y_train = np.eye(10)[y_train]
#     y_test = np.eye(10)[y_test]
#     x_valid = x_test
#     y_valid = y_test
#     return x_train, y_train, x_valid, y_valid, x_test, y_test

def get_fashion(permute=False):  #this MNIST is the fashion-mnist renamed
    from tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST", one_hot=True)

    if permute:
        perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    else:
        perm_mask = np.arange(784)

    x_train = list(np.expand_dims(mnist.train.images[:,perm_mask],-1))
    y_train = mnist.train.labels
    x_valid = list(np.expand_dims(mnist.validation.images[:,perm_mask],-1))
    y_valid = mnist.validation.labels
    x_test = list(np.expand_dims(mnist.test.images[:,perm_mask], -1))
    y_test = mnist.test.labels

    print("Train:Validation:Testing - %d:%d:%d" % (len(y_train), len(y_valid),
                                                   len(y_test)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test
