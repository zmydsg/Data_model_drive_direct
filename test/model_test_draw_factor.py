import pickle
import sys

import numpy as np
from train.findFuncAnswer import get_Equal_solution
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sea
import pandas as pd
from train.tempargs import args

def getSolution(factors, NumK, PDB, modelname):
    """
    read data in log file
    """
    delay_list, outage_list1 = [], []
    f = open(f"testlog\{modelname}_{PDB}_{NumK}_factor.pk", 'rb')
    data = pickle.load(f)
    for v, k in enumerate(data):
        delay_list.append(data[k][0]['delay'].item() if data[k][0]['delay'] else None)
        outage_list1.append(data[k][0]['pout'].item() if data[k][0]['pout'] else None)
   
    return delay_list, outage_list1

def draw_factor_graph(factors, NumKs, PDBs):
    project_path = "../"
    graph_path = project_path+"graph"


    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    GNN_through_HARQ_IR, GNN_out_HARQ_IR = getSolution(factors ,NumKs, PDBs, "HARQ-IR")
    GNN_through_HARQ_CC, GNN_out_HARQ_CC = getSolution(factors ,NumKs, PDBs, "HARQ-CC")
    GNN_through_HARQ, GNN_out_HARQ = getSolution(factors, NumKs, PDBs, "HARQ")
    
    if not os.path.exists('csv/factor_outage.csv') or not os.path.exists('csv/factor_through.csv'):
        outageListToMatlab = [GNN_out_HARQ, GNN_out_HARQ_CC, GNN_out_HARQ_IR]
        throughListToMatlab = [GNN_through_HARQ, GNN_through_HARQ_CC, GNN_through_HARQ_IR]
        df1 = pd.DataFrame(outageListToMatlab)
        df1.to_csv('csv/factor_outage.csv', na_rep='NaN', header=False, index=False)

        df2 = pd.DataFrame(throughListToMatlab)
        df2.to_csv("csv/factor_through.csv", na_rep='NaN', header=False, index=False)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(1)


    plt.plot(factors,GNN_through_HARQ, marker='o')
    plt.plot(factors, GNN_through_HARQ_CC, marker='+')
    plt.plot(factors, GNN_through_HARQ_IR, marker='*')

    # plt.plot(PDBs,Equal_through, linestyle='--', marker='*')
    plt.xlabel(u'$\\rho $', fontsize = 15)
    plt.ylabel('$ \\tau $', fontsize=15)
    plt.xticks(factors,fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])

    plt.figure(2)
    plt.semilogy(factors, GNN_out_HARQ,marker='o')
    plt.semilogy(factors, GNN_out_HARQ_CC, linestyle='--', marker='+')
    plt.semilogy(factors, GNN_out_HARQ_IR, marker='*')
    plt.xlabel(u'$\\rho $', fontsize=15)
    plt.ylabel('${p_{out\_asy,K}}$', fontsize=15)
    plt.xticks(factors,fontsize=10)
    plt.yticks([10e-3, 10e-4, 10e-5,],fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.show()

"""labels=["10DB","15DB","20DB","25DB"]"""
if __name__ == '__main__':

    factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92,0.94, 0.96, 0.98]
    Numks = 3
    PDBs = 15
    draw_factor_graph(factors, Numks, PDBs)









