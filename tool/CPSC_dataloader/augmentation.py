import os
import numpy as np
import matplotlib.pyplot as plt

LEAD_8=["I", "II",  "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_12=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def get_plots_1_channel( fake_ecg, epoch = None):

    fake_ecg_8_chs = fake_ecg.reshape(1, 1000)

    # fake_ecg_8_chs = ecg_filter(fake_ecg_8_chs, fs=100)
    # fake_ecg_8_chs = fake_ecg_8_chs.reshape(1, 2000)
    try:
        # Plot

        fig = plt.figure(figsize=(16, 4), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1]
        for i in range(len(fake_ecg_8_chs)):
            plt.subplot(1, 1, idx[i])
            plt.plot(fake_ecg_8_chs[i], color='green', label=str(i))
            plt.title(LEAD_8[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        #plt.savefig('./Plot_GengSi_oneC/plot_epoch{}'.format(epoch) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def get_plots_12_channel(fake_ecg):



    #fake_ecg_8_chs = fake_ecg.reshape(8, 2000)

    #ake_ecg_8_chs = ecg_filter(fake_ecg_8_chs, fs=100)
    try:
        # Plot

        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10,12]
        for i in range(len(fake_ecg)):
            plt.subplot(6, 2, idx[i])
            plt.plot(fake_ecg[i], color='green', label=str(i))
            plt.title(LEAD_12[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        #plt.savefig('./Plot/plot_CinC2011_epoch{}'.format(epoch) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False

def Am_Noisy(x, fs, amplitude):   # snr:信噪比

    print("Am")
    Am_all = []
    x_t = np.arange(0, x.shape[-1], 1)/5
    for i in range(x.shape[0]):

        Am_single = []
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            _, am_noisy = gen_am_noise(x_t, signal, fs, amplitude)
            Am_single.append(am_noisy)


        Am_single = np.array(Am_single)
        # get_plots_1_channel(signal, i)
        # get_plots_1_channel(Am_single, i)
        Am_all.append(Am_single)

    Am_all = np.array(Am_all)


    return Am_all



def Bw_Noisy(x, fs, amplitude):   # snr:信噪比

    print("Bw")
    Bw_all = []
    x_t = np.arange(0, x.shape[-1], 1)/50
    for i in range(x.shape[0]):

        Bw_single = []
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            _, bw_noisy = gen_bw_noise(x_t, signal, fs, amplitude)
            Bw_single.append(bw_noisy)


        Bw_single = np.array(Bw_single)
        # get_plots_1_channel(signal, i)
        # get_plots_1_channel(Bw_single, i)
        Bw_all.append(Bw_single)

    Bw_all = np.array(Bw_all)


    return Bw_all


def Fm_Noisy(x, fs, amplitude):   # snr:信噪比

    print("Fm")
    Fm_all = []
    x_t = np.arange(0, x.shape[-1], 1)/5
    for i in range(x.shape[0]):

        Fm_single = []
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            _, fm_noisy = gen_fm_noise(x_t, signal, fs, amplitude)
            Fm_single.append(fm_noisy)


        Fm_single = np.array(Fm_single)
        # get_plots_1_channel(signal, i)
        # get_plots_1_channel(Fm_single, i)
        Fm_all.append(Fm_single)

    Fm_all = np.array(Fm_all)


    return Fm_all



def Bw_Am_Fm_Noisy(x, fs_bw, amplitude_bw,fs_am, amplitude_am, fs_fm, amplitude_fm):   # snr:信噪比

    print("Bw_Am_Fm")
    Fm_all = []
    x_t = np.arange(0, x.shape[-1], 1)/5
    for i in range(x.shape[0]):

        Fm_single = []
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            bw_t, bw_noisy = gen_bw_noise(x_t, signal, fs_bw, amplitude_bw)
            am_t, am_noisy = gen_am_noise(bw_t, bw_noisy, fs_am, amplitude_am)
            fm_t, fm_noisy = gen_fm_noise(am_t, am_noisy, fs_fm, amplitude_fm)
            Fm_single.append(fm_noisy)


        Fm_single = np.array(Fm_single)
        #get_plots_8_channel(Fm_single, i)
        Fm_all.append(Fm_single)

    Fm_all = np.array(Fm_all)


    return Fm_all
# def Gamma_Noisy(x, snr):   # snr:信噪比
#
#     print('Gamma')
#     x_gamma = []
#     x_gamma_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         gamma = np.random.gamma(shape= 1, size = len(signal)) * np.sqrt(npower)
#
#         x_gamma.append(x[i] + gamma)
#         x_gamma_only.append(gamma)
#
#     x_gamma = np.array(x_gamma)
#     x_gamma_only = np.array(x_gamma_only)
#
#     return x_gamma, x_gamma.shape[-1]


def Gamma_Noisy(x, snr):   # snr:信噪比

    print("Gamma")
    Gamma_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        Gamma_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            gamma = np.random.gamma(shape= 1, size = len(signal)) * np.sqrt(npower)

            Gamma_single.append(x[i][j] + gamma)

        Gamma_single = np.array(Gamma_single)
        Gamma_all.append(Gamma_single)

    Gamma_all = np.array(Gamma_all)

    return Gamma_all

def Rayleign_Noisy(x, snr):   # snr:信噪比

    print("Rayleign")


    Rayleign_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        Rayleign_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            rayleign = np.random.rayleigh(size = len(signal)) * np.sqrt(npower)

            Rayleign_single.append(x[i][j] + rayleign)

        Rayleign_single = np.array(Rayleign_single)
        Rayleign_all.append(Rayleign_single)

    Rayleign_all = np.array(Rayleign_all)


    return Rayleign_all

# def Exponential_Noisy(x, snr):   # snr:信噪比
#
#     print("Exponential")
#     x_exponential = []
#     x_exponential_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         exponential = np.random.exponential(size = len(signal)) * np.sqrt(npower)
#
#         x_exponential.append(x[i] + exponential)
#         x_exponential_only.append(exponential)
#
#     x_exponential = np.array(x_exponential)
#     x_exponential_only = np.array(x_exponential_only)
#
#     return x_exponential, x_exponential.shape[-1]


def Exponential_Noisy(x, snr):   # snr:信噪比

    print("Exponential")


    exponential_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        exponential_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            exponential = np.random.exponential(size = len(signal)) * np.sqrt(npower)

            exponential_single.append(x[i][j] + exponential)

        exponential_single = np.array(exponential_single)
        exponential_all.append(exponential_single)

    exponential_all = np.array(exponential_all)


    return exponential_all

# def Uniform_Noisy(x, snr):   # snr:信噪比
#
#     print("Uniform")
#     x_uniform = []
#     x_uniform_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         uniform = np.random.uniform(size = len(signal)) * np.sqrt(npower)
#
#         x_uniform.append(x[i] + uniform)
#         x_uniform_only.append(uniform)
#
#     x_uniform = np.array(x_uniform)
#     x_uniform_only = np.array(x_uniform_only)
#
#     return x_uniform, x_uniform.shape[-1]



def Uniform_Noisy(x, snr):   # snr:信噪比

    print("Uniform")


    uniform_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        uniform_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            uniform = np.random.uniform(size = len(signal)) * np.sqrt(npower)

            uniform_single.append(x[i][j] + uniform)

        uniform_single = np.array(uniform_single)
        uniform_all.append(uniform_single)

    uniform_all = np.array(uniform_all)


    return uniform_all

# def Poisson_Noisy(x, snr):   # snr:信噪比
#
#     print("possion")
#     x_poisson = []
#     x_poisson_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         poisson = np.random.poisson(2, len(signal)) * np.sqrt(npower)
#
#         x_poisson.append(x[i] + poisson)
#         x_poisson_only.append(poisson)
#
#     x_poisson = np.array(x_poisson)
#     x_poisson_only = np.array(x_poisson_only)
#
#     return x_poisson, x_poisson.shape[-1]



def Poisson_Noisy(x, snr):   # snr:信噪比

    print("Poisson")


    poisson_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        poisson_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            poisson = np.random.poisson(2, len(signal)) * np.sqrt(npower)

            poisson_single.append(x[i][j] + poisson)

        poisson_single = np.array(poisson_single)
        poisson_all.append(poisson_single)

    poisson_all = np.array(poisson_all)


    return poisson_all

# def Gussian_Noisy(x, snr):   # snr:信噪比
#
#     print("Gussian")
#     x_gussian = []
#     x_gussian_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         gussian = np.random.randn(len(signal)) * np.sqrt(npower)
#
#         x_gussian.append(x[i] + gussian)
#         x_gussian_only.append(gussian)
#
#     x_gussian = np.array(x_gussian)
#     x_gussian_only = np.array(x_gussian_only)
#
#     return x_gussian, x_gussian.shape[-1]


def Gussian_Noisy(x, snr):   # snr:信噪比

    print("Gussian")


    gussian_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        gussian_single = []
        for j in range(x.shape[1]):

            signal = np.array(x[i][j])
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            gussian = np.random.randn(len(signal)) * np.sqrt(npower)

            gussian_single.append(x[i][j] + gussian)

        gussian_single = np.array(gussian_single)
        gussian_all.append(gussian_single)

    gussian_all = np.array(gussian_all)


    return gussian_all