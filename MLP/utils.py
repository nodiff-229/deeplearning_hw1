import numpy as np


def load_data(split):
    """Load dataset.

    :param split: a string specifying the partition of the dataset ('train' or 'test').
    :return: a (images, labels) tuple of corresponding partition.
    """

    images = np.load("./data/mnist_{}_images.npy".format(split))
    labels = np.load("./data/mnist_{}_labels.npy".format(split))
    return images, labels


def check_grad(calc_loss_and_grad):
    """Check backward propagation implementation. This is naively implemented with finite difference method.
    You do **not** need to modify this function.
    """

    def relative_error(z1, z2):
        return np.mean((z1 - z2) ** 2 / (z1 ** 2 + z2 ** 2))

    print('Gradient check of backward propagation:')

    # generate random test data
    x = np.random.rand(5, 15)
    y = np.random.rand(5, 3)
    # construct one hot labels
    y = y * (y >= np.max(y, axis=1, keepdims=True)) / np.max(y, axis=1, keepdims=True)

    # generate random parameters
    w1 = np.random.rand(15, 3)
    b1 = np.random.rand(3)
    w2 = np.random.rand(3, 3)
    b2 = np.random.rand(3)

    # calculate grad by backward propagation
    loss, db2, dw2, db1, dw1 = calc_loss_and_grad(x, y, w1, b1, w2, b2)

    # calculate grad by finite difference
    epsilon = 1e-5

    numeric_dw2 = np.zeros_like(w2)
    for i in range(w2.shape[0]):
        for j in range(w2.shape[1]):
            w2[i, j] += epsilon
            loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
            w2[i, j] -= epsilon
            numeric_dw2[i, j] = (loss_prime - loss) / epsilon
    print('Relative error of dw2', relative_error(numeric_dw2, dw2))

    numeric_db2 = np.zeros_like(b2)
    for i in range(db2.shape[0]):
        b2[i] += epsilon
        loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
        b2[i] -= epsilon
        numeric_db2[i] = (loss_prime - loss) / epsilon
    print('Relative error of db2', relative_error(numeric_db2, db2))

    numeric_dw1 = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w1[i, j] += epsilon
            loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
            w1[i, j] -= epsilon
            numeric_dw1[i, j] = (loss_prime - loss) / epsilon
    print('Relative error of dw1', relative_error(numeric_dw1, dw1))

    numeric_db1 = np.zeros_like(b1)
    for i in range(db1.shape[0]):
        b1[i] += epsilon
        loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
        b1[i] -= epsilon
        numeric_db1[i] = (loss_prime - loss) / epsilon

    print('Relative error of db1', relative_error(numeric_db1, db1))
    print('If you implement back propagation correctly, all these relative errors should be less than 1e-5.')

