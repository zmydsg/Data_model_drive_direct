import pickle
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# 获取项目根目录（假设脚本在 test/ 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 插入到最前确保优先级

from train.tempargs import *

graph_path = os.path.join(project_root, 'graphs')
csv_path = os.path.join(project_root, 'csv')

# 确保目标目录存在
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

def getSolution(factors, NumK, PDB, modelname):
    """
    read data in log file
    """
    delay_list, outage_list1 = [], []
    file_path = os.path.join("testlog", f"{modelname}_avg_{NumK}_factor.pk")  # 使用跨平台路径
    print(f"🔄 尝试加载文件路径: {os.path.abspath(file_path)}")  # 打印绝对路径
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    for v in data.values():
        delay_list.append(v[0]['delay'].item() if v[0]['delay'] else None)
        outage_list1.append(v[0]['pout'].item() if v[0]['pout'] else None)
   
    return delay_list, outage_list1

def draw_factor_graph(factors, NumKs, PDBs):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    GNN_through_HARQ_IR, GNN_out_HARQ_IR = getSolution(factors ,NumKs, PDBs, "HARQ-IR")
    GNN_through_HARQ_CC, GNN_out_HARQ_CC = getSolution(factors ,NumKs, PDBs, "HARQ-CC")
    GNN_through_HARQ, GNN_out_HARQ = getSolution(factors, NumKs, PDBs, "HARQ")
    
    if not os.path.exists(os.path.join(project_path, f'csv/factor_outage-NumK={NumKs}.csv')) or not os.path.exists(os.path.join(project_path,f'csv/factor_through-NumK={NumKs}.csv')):
        outageListToMatlab = [GNN_out_HARQ, GNN_out_HARQ_CC, GNN_out_HARQ_IR]
        throughListToMatlab = [GNN_through_HARQ, GNN_through_HARQ_CC, GNN_through_HARQ_IR]
        df1 = pd.DataFrame(outageListToMatlab)
        df1.to_csv(os.path.join(project_path,f'csv/factor_outage-NumK={NumKs}.csv'), na_rep='NaN', header=False, index=False)

        df2 = pd.DataFrame(throughListToMatlab)
        df2.to_csv(os.path.join(project_path,f"csv/factor_through-NumK={NumKs}.csv"), na_rep='NaN', header=False, index=False)

    # 保存不同 model_name 类型下的 rho 和 pout 值到 CSV 文件
    df_harq = pd.DataFrame({'rho': factors, 'pout': GNN_out_HARQ})
    df_harq.to_csv(os.path.join(csv_path, f'avg_rho_pout_HARQ_NumK={NumKs}.csv'), index=False)

    df_harq_cc = pd.DataFrame({'rho': factors, 'pout': GNN_out_HARQ_CC})
    df_harq_cc.to_csv(os.path.join(csv_path, f'avg_rho_pout_HARQ_CC_NumK={NumKs}.csv'), index=False)

    df_harq_ir = pd.DataFrame({'rho': factors, 'pout': GNN_out_HARQ_IR})
    df_harq_ir.to_csv(os.path.join(csv_path, f'avg_rho_pout_HARQ_IR_NumK={NumKs}.csv'), index=False)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(1, figsize=(10,8))

    plt.plot(factors,GNN_through_HARQ, marker='o')
    plt.plot(factors, GNN_through_HARQ_CC, marker='+')
    plt.plot(factors, GNN_through_HARQ_IR, marker='*')

    plt.xlabel(u'$\\rho $', fontsize = 15)
    plt.ylabel('$ \\tau $', fontsize=15)
    plt.xticks(factors,fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.savefig(os.path.join(graph_path, f'delays-NumK={NumKs}.jpg'))

    plt.figure(2, figsize=(10,8))
    plt.semilogy(factors, GNN_out_HARQ,marker='o')
    plt.semilogy(factors, GNN_out_HARQ_CC, linestyle='--', marker='+')
    plt.semilogy(factors, GNN_out_HARQ_IR, marker='*')
    plt.xlabel(u'$\\rho $', fontsize=15)
    plt.ylabel('${p_{out\\_asy,K}}$', fontsize=15)
    plt.xticks(factors,fontsize=10)
    plt.yticks([10e-3, 10e-4, 10e-5,],fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.savefig(os.path.join(graph_path, f'p_out-ask,K-NumK={NumKs}.jpg'))
    plt.show()

if __name__ == '__main__':
    factors = args.factors
    Numks = args.NumK
    PDBs = 15
    draw_factor_graph(factors, Numks, PDBs)

