import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE



# 重构效果可视化
def plot_rec(input, output, label, name, opt):
    # input:输入
    # output:输出
    # label:标签

    print(name + " start")

    fig, axs = plt.subplots(10, 8, dpi=100, figsize=(30, 30))
    axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    heat_rec = (np.array(output.tolist()) - np.array(input.tolist())) ** 2

    for i in [17]:
        ax_11 = axs[i].inset_axes([0.0, 0.26, 1, 0.74])
        if label.tolist()[i] == 0:
            color = 'blue'
            input_label = "Normality"
        else:
            color = 'red'
            input_label = "Anomaly"

        ax_11.plot(input.tolist()[i][0], color="blue", linestyle='-', linewidth=1, label="Input")
        ax_11.plot(output.tolist()[i][0], color="black", linestyle='--', linewidth=1, label="Ouput")

        ax_11.set_yticks([])
        if i == 17:
            ax_11.legend(loc='upper left', bbox_to_anchor=(-0.0, 1.5), ncol=3, fontsize=10)

        ax_13 = axs[i].inset_axes([0.0, 0.0, 1, 0.10], sharex=ax_11)
        heat_1 = (np.array(output.tolist()[i][0]) - np.array(input.tolist()[i][0])) ** 2
        heat_norm_1 = (heat_1 - np.min(heat_rec)) / (np.max(heat_rec) - np.min(heat_rec))
        heat_norm_1 = np.reshape(heat_norm_1, (1, -1))

        vmax = np.max(heat_norm_1)

        ax_13.imshow(heat_norm_1, cmap="jet", aspect="auto", vmin=0, vmax=vmax)
        ax_13.set_yticks([])

        ax_11.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_13.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax_13.text(-0.03, 0.5, "Rec", transform=ax_13.transAxes, fontsize=10, va='center', ha='right')

    # plt.show()

    plt.savefig(
        './{}/plot_rec/{}/{}/{}_{}_{}.svg'.format(opt.outf, opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
        transparent=False,
        bbox_inches='tight')
    plt.savefig(
        './{}/plot_rec/{}/{}/{}_{}_{}.pdf'.format(opt.outf, opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
        transparent=False,
        bbox_inches='tight')
    plt.close()
    print('Plot')

# 异常分数分布可视化
def plot_hist(scores, true_labels, opt, display=False):
    # scores:异常分数
    # true_labels:标签

    plt.figure()
    plt.grid(False)
    # plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-v0_8-bright')

    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    # hrange1 = (0.0005, 0.02)
    # hrange2 = (0, 0.02)
    # hrange = (min(scores), max(scores))
    hrange = (0, 0.3)

    # plt.hist(scores[idx_inliers], 50, facecolor=(0, 0.4, 1, 0.5),  # 浅绿色
    #          label="Normal", density=False, range=hrange, )
    # plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),  # 浅红色
    #          label="Abnormal", density=False, range=hrange)

    plt.hist(scores[idx_inliers], 50, facecolor=(82 / 255, 159 / 255, 119 / 255, 0.7),  # 浅绿色
             label="Normal", density=False, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(189 / 255, 74 / 255, 75 / 255, 0.7),  # 浅红色
             label="Abnormal", density=False, range=hrange)

    # plt.hist(scores[idx_inliers], 50, facecolor=(82 / 255, 159 / 255, 119 / 255, 0.7),  # 浅绿色
    #          label="Normal", density=False, range=hrange, edgecolor=(148 / 255, 0 / 255, 211 / 255, 0.7))
    # plt.hist(scores[idx_outliers], 50, facecolor=(189 / 255, 74 / 255, 75 / 255, 0.7),  # 浅红色
    #          label="Abnormal", density=False, range=hrange, edgecolor=(148 / 255, 0 / 255, 211 / 255, 0.7))

    plt.tick_params(labelsize=22)

    # xticks = [0, 0.0005, 0.001]  # 自定义横坐标的刻度位置
    # plt.xticks(xticks)

    plt.rcParams.update({'font.size': 22})
    plt.xlabel('AnomalyScore', fontsize=22)
    plt.ylabel('Count', fontsize=22)
    plt.legend(loc="upper right")

    # plt.ylim(0, 2200)

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_hist'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        # plt.savefig('{}/{}.png'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')


# 特征分布可视化
def plot_tsne_sns(latent, true_labels, opt, display=False):
    # latent = latent.cpu()
    # true_labels.cpu()

    pos = TSNE(n_components=2).fit_transform(latent)
    df = pd.DataFrame()
    df['x'] = pos[:, 0]
    df['y'] = pos[:, 1]
    # df['z'] = pos[:, 2]
    legends = list(range(10000))
    df['class'] = [legends[i] for i in true_labels]

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    cmap = sns.color_palette("Set2", 10)

    sns.lmplot(data=df,  # Data source
               x='x',  # Horizontal axis
               y='y',  # Vertical axis
               fit_reg=False,  # Don't fix a regression line
               hue="class",  # Set color,
               palette=cmap,
               legend=False,
               scatter_kws={"s": 25, 'alpha': 0.8})  # S marker size

    # ax.set_zlabel('pca-three')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    plt.xticks([])
    plt.yticks([])

    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()
    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')

def plot_tsne_3D(latent, label, opt, display=False):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 实例化 t-SNE 对象并将数据拟合到 3 维空间
    tsne = TSNE(n_components=3, random_state=0)
    X_3d = tsne.fit_transform(latent)

    # 将点按标签分组
    group1 = X_3d[label == 0]
    group2 = X_3d[label == 1]

    # 绘制 3D 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(group1[:, 0], group1[:, 1], group1[:, 2], c='blue', label='Group 1')
    ax.scatter(group2[:, 0], group2[:, 1], group2[:, 2], c='red', label='Group 2')
    ax.legend()

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne_3D'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')

# 心电图可视化
def plot_ecg_sample(signal, opt, i, display=False):
    # Plot
    fig = plt.figure(figsize=(30, 12), dpi=100)
    x = np.arange(0, 1000, 100)
    x_labels = np.arange(0, 10)

    plt.plot(signal, color='cadetblue')

    plt.xticks(x, x_labels)
    plt.xlabel('time (s)', fontsize=50)
    plt.ylabel('value (mV)', fontsize=50)

    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)

    fig.tight_layout()
    # plt.savefig("Plot_Hist/11k_{}".format(9999) + '.svg', bbox_inches='tight')
    if display:
        plt.show()
    else:
        save_dir = opt.outf
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.pdf'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')

def plot_frequency_sample(signal, opt, i, display=False):

    fft = np.fft.fft(signal)
    FFT_Power = np.abs(fft) ** 2

    # FFT中频率值为0-500 Hz,x轴频率范围
    freq = np.linspace(0, 500, 1000)

    plt.plot(freq, FFT_Power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')


    if display:
        plt.show()
    else:
        save_dir = opt.outf
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.pdf'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')

def plot_hist_hrv(scores, true_labels, opt, type, display=False):
    plt.figure()
    plt.grid(False)
    # plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'
    plt.style.use('seaborn-bright')  # 'seaborn-bright'

    idx_naf = (true_labels == 0)
    idx_af = (true_labels == 1)
    idx_paf = (true_labels == 2)

    hrange = (min(scores), max(scores))


    # plt.hist(scores[idx_naf], 50, facecolor=(82 / 255, 159 / 255, 119 / 255, 0.7),  # 浅绿色
    #          label="NAF", density=False, range=hrange)
    # plt.hist(scores[idx_af], 50, facecolor=(189 / 255, 74 / 255, 75 / 255, 0.7),  # 浅红色
    #          label="AF", density=False, range=hrange)
    # plt.hist(scores[idx_paf], 50, facecolor=(0 / 255, 128 / 255, 255 / 255, 0.7),  # 浅蓝色
    #          label="PAF", density=False, range=hrange)

    colors = {
        "NAF": "#E15759",  # 淡蓝色
        "AF": "#777777",  # 淡紫色
        "PAF": "#EDC949"  # 淡黄色EDC949
    }

    plt.hist(scores[idx_naf], bins=50, color=colors["NAF"], alpha=0.7, label="NAF", density=False, range=hrange)
    plt.hist(scores[idx_af], bins=50, color=colors["AF"], alpha=0.7, label="AF", density=False, range=hrange)
    plt.hist(scores[idx_paf], bins=50, color=colors["PAF"], alpha=0.7, label="PAF", density=False, range=hrange)

    plt.tick_params(labelsize=22)

    # xticks = [0, 0.0005, 0.001]  # 自定义横坐标的刻度位置
    # plt.xticks(xticks)

    plt.rcParams.update({'font.size': 22})
    plt.xlabel(type, fontsize=22)
    plt.ylabel('Count', fontsize=22)
    plt.legend(loc="upper right")

    # plt.ylim(0, 2200)

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_hist'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.svg'.format(save_dir, type), transparent=False, bbox_inches='tight')
        plt.close()
        print('Plot')

def plot_hrv(scores, true_labels, opt, type):

    idx_naf = (true_labels == 0)
    idx_af = (true_labels == 1)
    idx_paf = (true_labels == 2)

    naf = scores[idx_naf]
    af = scores[idx_af]
    paf = scores[idx_paf]

    sns.set(style='white')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax = [ax]
    for i in range(1):

        sns.distplot(naf, ax=ax[i], hist=False, kde_kws={'fill': True}, kde=True, color='green', label='NAF')
        ax[i].vlines(np.median(naf), 0, 0, color='green', linestyles='dashed')
        sns.distplot(af, ax=ax[i], hist=False, kde_kws={'fill': True}, kde=True, color='red', label='AF')
        ax[i].vlines(np.median(af), 0, 0, color='red', linestyles='dashed')
        sns.distplot(paf, ax=ax[i], hist=False, kde_kws={'fill': True}, kde=True, color='red', label='PAF')
        ax[i].vlines(np.median(paf), 0, 0, color='orange', linestyles='dashed')

    save_dir = '{}/plot_dist'.format(opt.outf)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig('{}/{}.svg'.format(save_dir, type), transparent=False, bbox_inches='tight')
    plt.close()
    print('Plot')


def plot_hist_hrv_p(scores, true_labels, opt, type, display=False):

    idx_naf = (true_labels == 0)
    idx_af = (true_labels == 1)
    idx_paf = (true_labels == 2)

    naf = scores[idx_naf]
    af = scores[idx_af]
    paf = scores[idx_paf]

    # 使用Seaborn绘制直方图和核密度曲线
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    plt.xlim(min(scores), 550)

    plt.rcParams.update({'font.size': 24})

    plt.ylabel('Count', fontsize=24)
    plt.xlabel('Value', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(loc="upper right")

    plt.text(0.5, 0.9, type,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=24)

    # plt.title(type, fontsize=16)

    binwidth = (max(scores) - min(scores)) / 70

    # 绘制第一个分布
    sns.histplot(naf, kde=False, label='NAF', color='green', binwidth=binwidth, alpha=0.7)

    # 绘制第二个分布
    sns.histplot(af, kde=False, label='AF', color='blue', binwidth=binwidth, alpha=0.7)

    # 绘制第三个分布
    sns.histplot(paf, kde=False, label='PAF', color='red', binwidth=binwidth, alpha=0.7)

    # 添加图例
    plt.legend()

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_hist_p'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.svg'.format(save_dir, type), transparent=False, bbox_inches='tight')
        plt.close()
        print('Plot')