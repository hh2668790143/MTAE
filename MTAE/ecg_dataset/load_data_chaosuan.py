import numpy as np


def load_chaosuan_data(opt):
    data_N_X = np.load(
        '/home/changhuihui/xindian_project/signal-quality-assessment-by-residual-recurrent-based-deep-learning-method-main/chaosuan_data/data_acc.npy')

    data_A_X = np.load(
        '/home/changhuihui/xindian_project/signal-quality-assessment-by-residual-recurrent-based-deep-learning-method-main/chaosuan_data/data_unacc.npy')

    np.random.seed(opt.seed)
    np.random.shuffle(data_N_X)
    np.random.shuffle(data_A_X)

    """测试用"""
    test_normal_X = np.load(
        '/home/changhuihui/xindian_project/signal-quality-assessment-by-residual-recurrent-based-deep-learning-method-main/chaosuan_data/data_acctest.npy')
    test_abnormal_X = np.load(
        '/home/changhuihui/xindian_project/signal-quality-assessment-by-residual-recurrent-based-deep-learning-method-main/chaosuan_data/data_unacctest.npy')

    train_X = data_N_X[:1800]
    train_X_a = data_A_X[:2500]

    val_normal_X = data_N_X[1800:]
    val_abnormal_X = data_A_X[2500:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X, train_X_a


if __name__ == '__main__':
    load_chaosuan_data()
