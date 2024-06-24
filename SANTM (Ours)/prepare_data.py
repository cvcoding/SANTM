import numpy as np
import random
from aux_code.ops import randomly_split_data
from aux_code.ops import one_hot_sequence

START_PERIOD = 0
END_PERIOD = 3
START_TARGET_PERIOD = 1
END_TARGET_PERIOD = 2
Num_Class = 2


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

    if dataset_name == 'add':
        x, y = get_add(n_data=50000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=50000, seq_len=seq_len)

    if dataset_name == 'freq':
        x, y = get_freq(30000, 1.0, seq_len, Num_Class, START_PERIOD, END_PERIOD, START_TARGET_PERIOD, END_TARGET_PERIOD)
        # return get_freq()

    train_idx, valid_idx, test_idx = randomly_split_data(
        y, test_frac=0.2, valid_frac=0.1)

    x_train = [x[i] for i in train_idx]
    y_train = y[train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_valid = y[valid_idx]
    x_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def generate_example(t, frequency, phase_shift):
    return np.cos(2 * np.pi * frequency * t + phase_shift)


def random_disjoint_interval(start, end, avoid_start, avoid_end):
    """
    Sample a value in [start, avoid_start] U [avoid_end, end] with uniform probability
    """
    val = random.uniform(start, end - (avoid_end - avoid_start))
    if val > avoid_start:
        val += (avoid_end - avoid_start)
    return val


def get_freq(n_data, sampling_period, signal_duration, n_classes, start_period, end_period,
    start_target_period, end_target_period):
    seq_length = int(signal_duration / sampling_period)

    n_elems = 1
    x = np.empty((n_data, seq_length, n_elems))
    y = np.empty((n_data, n_classes), dtype=np.int64)

    t = np.linspace(0, signal_duration - sampling_period, seq_length)

    for idx in range(int(n_data/2)):
        period = random.uniform(start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx, 0] = 1
        y[idx, 1] = 0
    for idx in range(int(n_data/2), n_data):
        period = random_disjoint_interval(start_period, end_period,
                                          start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx, 0] = 0
        y[idx, 1] = 1
    return x, y


def get_add3(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 3))
    x[:,:,0] = np.random.uniform(low=-0.5, high=0.5, size=(n_data, seq_len))
    inds = np.random.randint(seq_len/10, size=(n_data, 3))
    inds[:, 1] += (seq_len*2)//5
    inds[:, 2] += (seq_len * 4) // 5
    for i in range(n_data):
        x[i,inds[i,0],1] = 1.0
        x[i,inds[i,1],1] = 1.0
        x[i,inds[i,2],1] = 1.0

    y = (x[:,:,0] * x[:,:,1]).sum(axis=1)
    y = np.reshape(y, (n_data, 1))
    return x, y


def get_add(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 2))
    x[:, :, 0] = np.random.uniform(low=-0.5, high=0.5, size=(n_data, seq_len))
    inds = np.random.randint(seq_len/10, size=(n_data, 2))
    inds[:, 1] += (seq_len*4)//5
    for i in range(n_data):
        x[i, inds[i, 0], 1] = 1.0
        x[i, inds[i, 1], 1] = 1.0
    y = (x[:, :, 0] * x[:, :, 1]).sum(axis=1)
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

    n_train, n_valid, n_test = [30000, 5000, 15000]
    x_train = list(x[:n_train])
    y_train = y[:n_train]
    x_valid = list(x[n_train:n_train+n_valid])
    y_valid = y[n_train:n_train+n_valid]
    x_test = list(x[-n_test:])
    y_test = y[-n_test:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_mnist(permute=False):
    from tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST", one_hot=True)

    if permute:
        perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    else:
        perm_mask = np.arange(784)

    x_train = list(np.expand_dims(mnist.train.images[:,perm_mask],-1))
    # x_train = [np.random.uniform(low=0, high=1, size=(784, 129)) for _ in range(5500)]
    y_train = mnist.train.labels
    x_valid = list(np.expand_dims(mnist.validation.images[:,perm_mask],-1))
    y_valid = mnist.validation.labels
    x_test = list(np.expand_dims(mnist.test.images[:,perm_mask], -1))
    y_test = mnist.test.labels

    print("Train:Validation:Testing - %d:%d:%d" % (len(y_train), len(y_valid),
                                                   len(y_test)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test
