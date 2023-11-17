# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/12/19 22:33
import matplotlib.pyplot as plt
import numpy as np

def draw_loss(Loss_list,dir):
    try:
        plt.cla()
        x1 = range(1, len(Loss_list)+1)
        y1 = Loss_list

        plt.title('Train loss vs. epoches', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoches', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig(dir)
        # plt.savefig("./lossAndacc/Train_loss.png")
        plt.show()
        plt.close()
    except Exception as e:
        print(e)
