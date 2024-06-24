from __future__ import print_function
import numpy as np
from aux_code.ops import randomly_split_data
from aux_code.ops import one_hot_sequence
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
from sklearn import preprocessing
import scipy.io as sio
# import tensorflow as tf
import random


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

    if dataset_name == 'voice':
        return get_voice()

    if dataset_name == 'add':
        x, y = get_add(n_data=50000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=150000, seq_len=seq_len)


def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def log_and_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp = np.log(data[0][i])
        temp = (temp-mean)/std
        #plt.imshow(temp.T,origin='lower')
        log_norm_data.append(temp)
    return np.array(log_norm_data, dtype=object)


def shuffle_data(AC, y, trainNum):
    index = list(range(trainNum))
    random.shuffle(index)
    AC = AC[index]
    y = y[index]
    return AC, y



def load_voice(sift_set, datainfo, num):
    TRAIN1 = sio.loadmat(sift_set)
    BC1 = TRAIN1['STFT_bc']
    dataInfo1 = sio.loadmat(datainfo)
    BC_mean1, BC_std1 = dataInfo1['log_STFT_bc_mean'], dataInfo1['log_STFT_bc_var']
    BC1 = log_and_normalize(BC1, BC_mean1, BC_std1)
    BC1 = np.array(BC1)
    n_data = BC1.shape[0]
    n_classes = 5
    Y1 = np.empty((n_data, n_classes), dtype=np.int64)
    if num==1:
        for idx in range(int(n_data)):
            Y1[idx, 0] = 1
            Y1[idx, 1] = 0
            Y1[idx, 2] = 0
            Y1[idx, 3] = 0
            Y1[idx, 4] = 0
            # Y1[idx, 5] = 0
    elif num==2:
        for idx in range(int(n_data)):
            Y1[idx, 0] = 0
            Y1[idx, 1] = 1
            Y1[idx, 2] = 0
            Y1[idx, 3] = 0
            Y1[idx, 4] = 0
            # Y1[idx, 5] = 0
    elif num==3:
        for idx in range(int(n_data)):
            Y1[idx, 0] = 0
            Y1[idx, 1] = 0
            Y1[idx, 2] = 1
            Y1[idx, 3] = 0
            Y1[idx, 4] = 0
            # Y1[idx, 5] = 0
    elif num==4:
        for idx in range(int(n_data)):
            Y1[idx, 0] = 0
            Y1[idx, 1] = 0
            Y1[idx, 2] = 0
            Y1[idx, 3] = 1
            Y1[idx, 4] = 0
            # Y1[idx, 5] = 0
    elif num==5:
        for idx in range(int(n_data)):
            Y1[idx, 0] = 0
            Y1[idx, 1] = 0
            Y1[idx, 2] = 0
            Y1[idx, 3] = 0
            Y1[idx, 4] = 1
            # Y1[idx, 5] = 0
    # elif num==6:
    #     for idx in range(int(n_data)):
    #         Y1[idx, 0] = 0
    #         Y1[idx, 1] = 0
    #         Y1[idx, 2] = 0
    #         Y1[idx, 3] = 0
    #         Y1[idx, 4] = 0
    #         Y1[idx, 5] = 1

    return BC1, Y1


def get_voice():

    sift_set = './data/8female/f001_STFT_TRAINSET'
    datainfo = './data/8female/f001_datainfo.mat'
    num = 1
    BC1, Y1 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f002_STFT_TRAINSET'
    datainfo = './data/8female/f002_datainfo.mat'
    num = 2
    BC2, Y2 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f003_STFT_TRAINSET'
    datainfo = './data/8female/f003_datainfo.mat'
    num = 3
    BC3, Y3 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f004_STFT_TRAINSET'
    datainfo = './data/8female/f004_datainfo.mat'
    num = 4
    BC4, Y4 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f005_STFT_TRAINSET'
    datainfo = './data/8female/f005_datainfo.mat'
    num = 5
    BC5, Y5 = load_voice(sift_set, datainfo, num)

    # sift_set = '8female/f006_STFT_TRAINSET'
    # datainfo = '8female/f006_datainfo.mat'
    # num = 6
    # BC6, Y6 = load_voice(sift_set, datainfo, num)

    BC = np.concatenate([BC1, BC2, BC3, BC4, BC5], axis=0)
    Ytr = np.concatenate([Y1, Y2, Y3, Y4, Y5], axis=0)

    n_data1 = BC.shape[0]
    BC, Ytr = shuffle_data(BC, Ytr, n_data1)

#################test####################################

    sift_set = './data/8female/f001_STFT_TESTSET'
    datainfo = './data/8female/f001_datainfo.mat'
    num =1
    BCtest1, Ytest1 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f002_STFT_TESTSET'
    datainfo = './data/8female/f002_datainfo.mat'
    num =2
    BCtest2, Ytest2 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f003_STFT_TESTSET'
    datainfo = './data/8female/f003_datainfo.mat'
    num = 3
    BCtest3, Ytest3 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f004_STFT_TESTSET'
    datainfo = './data/8female/f004_datainfo.mat'
    num = 4
    BCtest4, Ytest4 = load_voice(sift_set, datainfo, num)

    sift_set = './data/8female/f005_STFT_TESTSET'
    datainfo = './data/8female/f005_datainfo.mat'
    num = 5
    BCtest5, Ytest5 = load_voice(sift_set, datainfo, num)

    # sift_set = '8female/f006_STFT_TESTSET'
    # datainfo = '8female/f006_datainfo.mat'
    # num = 6
    # BCtest6, Ytest6 = load_voice(sift_set, datainfo, num)

    BCte = np.concatenate([BCtest1, BCtest2, BCtest3, BCtest4, BCtest5], axis=0)
    Yte = np.concatenate([Ytest1, Ytest2, Ytest3, Ytest4, Ytest5], axis=0)
    n_data2 = BCte.shape[0]
    BCte, Yte = shuffle_data(BCte, Yte, n_data2)

    ###########the same long sequences################

    featdim = BCtest1[0].shape[-1]
    BCtr_final = np.zeros((n_data1, 1280, featdim))
    Masking_tr = np.zeros((n_data1, 1280, featdim))
    for i in range(BC.shape[0]):
            BCtr_final[i, :BC[i].shape[0], :] = BC[i]
            Masking_tr[i, :BC[i].shape[0], :] = np.ones(BC[i].shape)
    BCte_final = np.zeros((n_data2, 1280, featdim))
    Masking_te = np.zeros((n_data2, 1280, featdim))
    for i in range(BCte.shape[0]):
            BCte_final[i, :BCte[i].shape[0], :] = BCte[i]
            Masking_te[i, :BCte[i].shape[0], :] = np.ones(BCte[i].shape)
    return BCtr_final, Ytr, Masking_tr, BCte_final, Yte, Masking_te

