import matplotlib

matplotlib.use('Agg')

import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import os


def create_logdir(model, dataset):
    """ Directory to save training logs, weights, biases, etc."""
    return "../embeddings/{}/{}".format(model, dataset)


def create_scoredir(model, dataset):
    """ Directory to save training logs, weights, biases, etc."""
    return "../score/{}/{}/0".format(model, dataset)


def create_scoredirCoCo(model, dataset):
    """ Directory to save training logs, weights, biases, etc."""
    return "../score/{}/{}".format(model, dataset)


def plot_distribution(x1, x2, label1, label2, save_dir, opt):
    sns.set(style='white')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax = [ax]
    for i in range(1):
        # plot vi frequency distribution
        sns.distplot(x2, ax=ax[i], hist=False, kde_kws={'fill': True}, kde=True, color='green', label='normal')
        ax[i].vlines(np.median(x2), 0, 0, color='green', linestyles='dashed')
        sns.distplot(x1, ax=ax[i], hist=False, kde_kws={'fill': True}, kde=True, color='red', label='abnormal')
        ax[i].vlines(np.median(x1), 0, 0, color='red', linestyles='dashed')
        # ax[i].set_ylim(0, 4)
        # ax[i].set_xlim(-4,0)
        # ax[i].set_title('Data', size=15)
        # ax[i].set_xlabel('Anomaly Score', size=15)
        # ax[i].set_ylabel('Frequency', size=15)
    # ax[i].legend(loc='upper left', fontsize=15)
    plt.savefig(save_dir + "/dist_{}".format(opt.seed) + '.eps', dpi=330, bbox_inches='tight')
    plt.savefig(save_dir + "/dist_{}".format(opt.seed) + '.png', bbox_inches='tight')
    fig.tight_layout()


def plot_dist(X1, X2, label1, label2, save_dir):
    assert save_dir is not None
    f = plt.figure()
    ax = f.add_subplot(111)

    # bins = np.linspace(0, 1, 50)
    # _,bins=ax.hist(X1,bins=50)
    # print(bins)
    #
    # if logscale:
    _, bins, _ = ax.hist(X2, 50, facecolor=(0, 0.4, 1, 0.5),  # 浅绿色
                         label=label2, density=False, range=[0, 1])
    _, bins, _ = ax.hist(X1, 50, facecolor=(1, 0, 0, 0.5),  # 浅红色
                         label=label1, density=False, range=[0, 1])
    #     bins = np.logspace(np.log10(bins[0]), np.log10(bins[1]), len(bins))
    # _ ,bins, _= ax.hist(X2, bins=50, range=[0,1],alpha=0.3,facecolor=(0,0.5,1,0.5),label=label2)
    # _, bins, _ = ax.hist(X1, bins=bins,alpha=0.3,facecolor=(1,0,0,0.5), label=label1)

    # ax.set_yticks([0,20,40,60,80,100])
    ax.legend(fontsize=16)

    f.savefig(os.path.join(save_dir, "dist_zz" + ".png"))
    plt.savefig(os.path.join(save_dir, "dist_zz" + ".svg"), bbox_inches='tight', dpi=600)
    # #log scale figure
    # f_log=plt.figure()
    # ax_log=f_log.add_subplot(111)
    #
    # log_bins=np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
    # _ =ax_log.hist(X1, bins=log_bins, range=[0,1],alpha=0.3,density=True,color='r',label=label1)
    # _ = ax_log.hist(X2, bins=log_bins,density=True, alpha=0.3,  color='b', label=label2)
    # # ax_log.set_yticks([])
    #
    # ax_log.legend()
    # ax_log.set_xscale('log')
    # ax_log.set_xticks([round(x,2) for x in log_bins[::5]])
    # ax_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax_log.set_xticklabels([round(x,2) for x in log_bins[::5]], rotation=45)
    # f_log.savefig(os.path.join(save_dir,"logdist"+label1+label2+".png"))
    plt.close()


def norm(score):
    return (score - score.min()) / (score.max() - score.min())


if __name__ == '__main__':
    import numpy as np

    foo = np.random.normal(loc=1, size=100)  # a normal distribution
    bar = np.random.normal(loc=-1, size=10000)  # a normal distribution
    max_val = max(np.max(foo), np.max(bar))
    min_val = min(np.min(foo), np.min(bar))
    foo = (foo - min_val) / (max_val - min_val)
    bar = (bar - min_val) / (max_val - min_val)

    DATA = ['cora', 'citeseer', 'Flickr', 'ACM', 'disney', 'reddit', 'weibo']  # ,'BlogCatalog']
    models = ['SVDAE', 'AnomalyDAE', 'ComGA', 'Conad', 'Dominant']
    # 'hopgraph'
    # dataset = 'citeseer'#'cora'
    for model in models:
        for dataset_name in DATA:

            score_dir = create_scoredir(model, dataset_name)
            coco_dir = create_scoredirCoCo('CoCo', dataset_name)
            vis_dir = os.path.sep.join(['../figure/AnomalyCA/', 'corr1', '{}'.format(model), '{}'.format(dataset_name)])

            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            if model in ['CoCo']:
                corr = np.load(os.path.sep.join([coco_dir, 'score.npy']))
            else:
                corr = np.load(os.path.sep.join([score_dir, 'score.npy']))
            if model in ['hopgraph']:
                labels = np.load(os.path.sep.join([score_dir, 'label.npy']))
            elif model in ['CoCo']:
                labels = np.load(os.path.sep.join([coco_dir, 'label.npy']))
            else:
                labels = np.load(os.path.sep.join([score_dir, 'label.txt.npy']))
            max_val = max(np.max(corr), np.max(labels))
            min_val = min(np.min(corr), np.min(labels))
            # corr = norm(corr)
            normal = corr[np.where(labels == 0)]
            # normal =  np.where(normal > 0.2, 0.19, normal)
            abnormal = corr[np.where(labels != 0)]
            num = len(abnormal)
            normal = np.sort(corr[np.where(labels == 0)])
            num_normal = len(normal)
            # index = np.random.randint(0,num_normal,size=[num])
            normal = normal[:num_normal]
            # abnormal = np.where(n:ormal < 0.2, 0.21, abnormal)
            max_val = max(np.max(normal), np.max(abnormal))
            min_val = min(np.min(normal), np.min(abnormal))
            normal = (normal - min_val) / (max_val - min_val)
            abnormal = (abnormal - min_val) / (max_val - min_val)
            plot_distribution(abnormal, normal, "Abnormal", "Normal", vis_dir)
            plot_dist(abnormal, normal, "Abnormal", "Normal", vis_dir)
