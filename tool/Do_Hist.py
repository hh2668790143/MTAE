import os
import random

from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# rom ProcessingData import load_data


def do_hist(scores, true_labels, opt, display=False):
    plt.figure()
    plt.grid(False)
    plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'
    # plt.style.use('seaborn-bright')  # 'seaborn-bright'

    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    # hrange1 = (0.0005, 0.02)
    # hrange2 = (0, 0.02)
    hrange = (min(scores), max(scores))
    # hrange = (min(scores), 0.5)
    # hrange=(0.0015,0.0075)
    # hrange=(min(scores),0.05)
    # plt.hist(scores[idx_inliers], 50, facecolor='black',
    #          label="Normal samples", density=False, range=hrange)
    # plt.hist(scores[idx_outliers], 50, facecolor='silver',
    #          label="Anomalous samples", density=False, range=hrange)
    # plt.hist(scores[idx_inliers], 50, facecolor=(0, 0, 0, 1),
    #          label="Normal samples", density=True, range=hrange)
    # plt.hist(scores[idx_outliers], 50, facecolor=(0.5, 0.5, 0.5, 0.5),
    #          label="Anomalous samples", density=True, range=hrange)
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 0.4, 1, 0.5),  # 浅绿色
             label="Normal", density=False, range=hrange, )
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),  # 浅红色
             label="Abnormal", density=False, range=hrange)

    plt.tick_params(labelsize=22)
    # ax = plt.gca()
    #
    # for i in ['top', 'right', 'bottom', 'left']:
    #     ax.spines[i].set_visible(False)
    # plt.title("Distribution of the anomaly score")
    # plt.grid()
    plt.rcParams.update({'font.size': 22})
    plt.xlabel('AnomalyScore', fontsize=22)
    plt.ylabel('Count', fontsize=22)
    plt.legend(loc="upper right")

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}_{}.svg'.format(save_dir, str(opt.normal_idx), str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')


# do_hist(data, y_test, directory='./0/', dataloader='alad', random_seed='raw_1', display=False)


if __name__ == '__main__':
    # name = 'RDF'
    # x_data = ['1', '2', '3']
    # y_data = [0.8238, 0.7943, 0.7315]
    # y2_data = [0.8919, 0.8639, 0.8001]
    # name = 'SKA'
    # x_data = ['1', '2', '3']
    # y_data = [0.838, 0.7981, 0.7961]
    # y2_data = [0.8943, 0.8687, 0.8654]
    name = 'FST'
    x_data = ['1', '2']
    y_data = [0.9758, 0.9697]
    y2_data = [0.9755, 0.9529]

    x_width = range(0, len(x_data))
    x2_width = [i + 0.2 for i in x_width]

    p1 = plt.bar(x_width, y_data, lw=0.2, fc="r", width=0.2, label='AUC')
    p2 = plt.bar(x2_width, y2_data, lw=0.2, fc="b", width=0.2, label='AP')
    # plt.bar_label(p1, label_type='center')
    # plt.bar_label(p1, label_type='center')
    plt.xticks(range(0, 2), x_data)
    plt.title(name)
    plt.ylabel("AUC/AP")
    plt.xlabel("Class")
    plt.legend(['AUC', 'AP'], loc='upper right')

    plt.show()
    plt.savefig('./pdf/{}.pdf'.format(name),
                transparent=False,
                bbox_inches='tight')

    plt.close()
    print('Plot')
