import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_plots_1_channel(data):
    try:
        fig = plt.figure(figsize=(16, 4), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        plt.plot(data, color='green', label="ECG")
        plt.title("I" + ": ", fontsize=16)
        plt.xticks(x, x_labels)
        plt.xlabel('time (s)', fontsize=16)
        plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        # plt.savefig('./Plot_GengSi_oneC/plot_epoch{}'.format(epoch) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False

# data_file = "/data/ECG/process/ALL_100/YXDECG_10000_data_FangChan.pickle"
# data_file = "/home/luweikai/data/DECG_cpsc_fangchan2/train/data_10_normal.pickle"
# data_file = "/home/luweikai/data/DECG_cpsc_fangchan/91/train/data_8_fangchan.pickle"
data_file = "/home/luweikai/data/DECG_cpsc_fangchan/training_I/data_0_1_normal.pickle"
data = []
with open(data_file, 'rb') as handle1:
    x = pickle.load(handle1)
    data.extend(x)

data = data[5]
data = data[:,:1000][0]
get_plots_1_channel(data)
# plt.figure()
# plt.rcParams['figure.figsize'] = (10.0, 5.0)
# plt.plot(range(0, data.shape[-1]), data[0], label='I', linewidth=3)
# plt.xticks(range(0, data.shape[-1], 5))
# plt.legend()
# plt.show()