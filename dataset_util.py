import os
import h5py
import numpy as np


def load_dataset():
    data_base_path = '/home/prantik/PycharmProjects/SignLanguage/datasets'
    train_data_path = os.path.join(data_base_path, 'train_signs.h5')
    test_data_path = os.path.join(data_base_path, 'test_signs.h5')

    train_data = h5py.File(train_data_path, 'r')
    test_data = h5py.File(test_data_path, 'r')

    train_data_x = train_data['train_set_x'][:] / 255
    train_data_y = train_data['train_set_y'][:]

    test_data_x = test_data['test_set_x'][:] / 255
    test_data_y = test_data['test_set_y'][:]

    y_classes = test_data['list_classes'][:]

    return train_data_x, train_data_y, test_data_x, test_data_y, y_classes


def get_minibatches(train_set_x, train_set_y, batch_size=32):
    num_tr_samples = train_set_x.shape[0]
    # shuffle the available data
    np.random.seed(1)
    p = np.random.permutation(num_tr_samples)

    shuffled_x = train_set_x[p]
    shuffled_y = train_set_y[p]

    mini_batches = list()

    num_full_batches = int(num_tr_samples / batch_size)

    for i in range(num_full_batches):
        mini_x = shuffled_x[i * batch_size : (i + 1) * batch_size, :, :, :]
        mini_y = shuffled_y[i * batch_size : (i + 1) * batch_size]
        mini_batches.append((mini_x, mini_y))

    if num_tr_samples % batch_size != 0:
        mini_x = shuffled_x[num_full_batches * batch_size:, :, :, :]
        mini_y = shuffled_y[num_full_batches * batch_size:]
        mini_batches.append((mini_x, mini_y))

    return mini_batches


if __name__ == '__main__':
    train_data_x, train_data_y, test_data_x, test_data_y, y_classes = load_dataset()
    mini_batches = get_minibatches(train_data_x, train_data_y)

    print()
