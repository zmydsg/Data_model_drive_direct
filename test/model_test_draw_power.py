import pickle
import numpy as np
from train.findFuncAnswer import get_Equal_solution
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sea
import pandas as pd
from args import args


def getSolution(NumK, modelname):
    """
    read data in log file
    """
    delay_list, outage_list1 = [], []
    f = open(f"testlog\{modelname}_{NumK}_power.pk", 'rb')
    data = pickle.load(f)

    for v, k in enumerate(data):
        delay_list.append(data[k][0]['delay'].item() if data[k][0]['delay'] else None)
        outage_list1.append(data[k][0]['pout'].item() if data[k][0]['pout'] else None)
    f.close()
    print(f"modelname:{modelname},delay_list:{delay_list}")
    print(f"outage_list1:{outage_list1}")
    return delay_list, outage_list1


def draw_power_graph(NumKs, PDBs):
    """
    POWER
    """

    print(f"PDBs:{PDBs}")

    GNN_delay_HARQ, GNN_out_HARQ = getSolution(NumKs, "HARQ")
    GNN_delay_HARQ_CC, GNN_out_HARQ_CC = getSolution(NumKs, "HARQ-CC")
    GNN_delay_HARQ_IR, GNN_out_HARQ_IR = getSolution(NumKs, "HARQ-IR")
    print("="*10)
    print(f"GNN_out_HARQ_IR:{GNN_out_HARQ_IR}")
    print("=" * 10)
    if not os.path.exists('csv/power_outage.csv') or not os.path.exists('csv/power_through.csv'):
        outageListToMatlab = [GNN_out_HARQ, GNN_out_HARQ_CC, GNN_out_HARQ_IR]
        throughListToMatlab = [GNN_delay_HARQ, GNN_delay_HARQ_CC, GNN_delay_HARQ_IR]
        df1 = pd.DataFrame(outageListToMatlab)
        df1.to_csv('csv/power_outage.csv', na_rep='NaN', header=False, index=False)

        df2 = pd.DataFrame(throughListToMatlab)
        df2.to_csv("csv/power_through.csv", na_rep='NaN', header=False, index=False)

    # GNN_through = np.array(GNN_through).reshape(len2,len1)
    # GNN_out = np.array(GNN_out, dtype=float).reshape(len2,len1)
    # print("GNN",GNN_through,"\t",GNN_out)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(1)

    plt.plot(PDBs,GNN_delay_HARQ, marker='o')
    plt.plot(PDBs, GNN_delay_HARQ_CC, marker='+')
    plt.plot(PDBs, GNN_delay_HARQ_IR, marker='*')

    # plt.plot(PDBs,Equal_through, linestyle='--', marker='*')
    plt.xlabel(u'$\\bar p$ (dB)', fontsize = 15)
    plt.ylabel('$\eta $ (bps/Hz)', fontsize=15)
    plt.xticks(PDBs,fontsize=10)
    plt.yticks([0.045, 0.05, 0.055, 0.06 ,0.065, 0.070, 0.075, 0.08],fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])

    plt.figure(2)
    plt.semilogy(PDBs, GNN_out_HARQ,marker='o')
    plt.semilogy(PDBs, GNN_out_HARQ_CC, linestyle='--', marker='+')
    plt.semilogy(PDBs, GNN_out_HARQ_IR, marker='*')
    print(np.all(GNN_out_HARQ_IR<np.array([10e-3])))
    plt.xlabel(u'$\\bar p$ (dB)', fontsize=15)
    plt.ylabel('${p_{out\_asy,K}}$', fontsize=15)
    plt.xticks(PDBs,fontsize=10)
    plt.yticks([10e-3, 10e-4, 10e-5, 10e-6, 10e-7, 10e-8],fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.show()


"""labels=["10DB","15DB","20DB","25DB"]"""
if __name__ == '__main__':
    #draw_loss_graph()
    PDBs = [10, 12, 14, 16, 18, 20, 22, 24]
    Numks = 3
    # PDBs = [i for i in np.arange(start=10, stop=26, step=2)]

    draw_power_graph(Numks, PDBs)






