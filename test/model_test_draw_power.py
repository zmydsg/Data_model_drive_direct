import pickle
import numpy as np
import sys
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

def getSolution(NumK, modelname, base=''):
    """
    read data in log file
    """
    delay_list, outage_list1 = [], []
    file_path = os.path.join("testlog", f"{modelname}_{NumK}_power{base}.pk")
    print(f"🔄 尝试加载文件路径: {os.path.abspath(file_path)}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    for v in data.values():
        delay_list.append(v[0]['delay'].item() if v[0]['delay'] else None)
        outage_list1.append(v[0]['pout'].item() if v[0]['pout'] else None)
    print(f"modelname:{modelname}, delay_list:{delay_list}")
    print(f"outage_list1:{outage_list1}")
    return delay_list, outage_list1

def draw_auto_pdb_power_graph(NumKs, PDBs):
    """
    POWER
    """
    print(f"PDBs:{PDBs}")

    GNN_delay_HARQ, GNN_out_HARQ = getSolution(NumKs, "HARQ", '_avg')
    GNN_delay_HARQ_CC, GNN_out_HARQ_CC = getSolution(NumKs, "HARQ-CC", '_avg')
    GNN_delay_HARQ_IR, GNN_out_HARQ_IR = getSolution(NumKs, "HARQ-IR", '_avg')

    if not os.path.exists(os.path.join(project_root, f'csv/power_outage-NumK={NumKs}_avg.csv')) or not os.path.exists(os.path.join(project_root, f'csv/power_through-NumK={NumKs}_avg.csv')):
        outageListToMatlab = [GNN_out_HARQ, GNN_out_HARQ_CC, GNN_out_HARQ_IR]
        throughListToMatlab = [GNN_delay_HARQ, GNN_delay_HARQ_CC, GNN_delay_HARQ_IR]
        df1 = pd.DataFrame(outageListToMatlab)
        df1.to_csv(os.path.join(project_root, f'csv/power_outage-NumK={NumKs}_avg.csv'), na_rep='NaN', header=False, index=False)

        df2 = pd.DataFrame(throughListToMatlab)
        df2.to_csv(os.path.join(project_root, f"csv/power_through-NumK={NumKs}_avg.csv"), na_rep='NaN', header=False, index=False)

    # 保存不同 model_name 类型下的 pout 和 bar p 值到 CSV 文件
    df_harq = pd.DataFrame({'bar_p': PDBs, 'pout': GNN_out_HARQ})
    df_harq.to_csv(os.path.join(csv_path, f'bar_p_pout_HARQ_NumK={NumKs}_avg.csv'), index=False)

    df_harq_cc = pd.DataFrame({'bar_p': PDBs, 'pout': GNN_out_HARQ_CC})
    df_harq_cc.to_csv(os.path.join(csv_path, f'bar_p_pout_HARQ_CC_NumK={NumKs}_avg.csv'), index=False)

    df_harq_ir = pd.DataFrame({'bar_p': PDBs, 'pout': GNN_out_HARQ_IR})
    df_harq_ir.to_csv(os.path.join(csv_path, f'bar_p_pout_HARQ_IR_NumK={NumKs}_avg.csv'), index=False)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(1, figsize=(10,8))

    plt.plot(PDBs, GNN_delay_HARQ, marker='o')
    plt.plot(PDBs, GNN_delay_HARQ_CC, marker='+')
    plt.plot(PDBs, GNN_delay_HARQ_IR, marker='*')

    plt.xlabel(u'$\\bar p$ (dB)', fontsize=15)
    plt.ylabel('$\eta $ (bps/Hz)', fontsize=15)
    plt.xticks(PDBs, fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.savefig(os.path.join(graph_path, f'eta(bpsHz)-NumK={NumKs}_avg.jpg'))

    plt.figure(2, figsize=(10,8))
    plt.semilogy(PDBs, GNN_out_HARQ, marker='o')
    plt.semilogy(PDBs, GNN_out_HARQ_CC, linestyle='--', marker='+')
    plt.semilogy(PDBs, GNN_out_HARQ_IR, marker='*')
    print(np.all(GNN_out_HARQ_IR < np.array([10e-3])))
    plt.xlabel(u'$\\bar p$ (dB)', fontsize=15)
    plt.ylabel('${p_{out\_asy,K}}$', fontsize=15)
    plt.xticks(PDBs, fontsize=10)
    plt.legend([u"Type-I HARQ", u"HARQ-CC", u"HARQ-IR"])
    plt.savefig(os.path.join(graph_path, f'bar-p_out-asy,K-NumK={NumKs}_avg.jpg'))
    plt.show()

if __name__ == '__main__':
    PDBs = args.PDBs
    Numks = args.NumK
    draw_auto_pdb_power_graph(Numks, PDBs)






