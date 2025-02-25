import sys
import os

# 获取项目根目录（假设脚本在 test/ 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 插入到最前确保优先级

# 确保 train 模块所在的路径在 sys.path 中
train_path = os.path.join(project_root, 'train')
if train_path not in sys.path:
    sys.path.insert(0, train_path)

import torch
from train.utils import getfactordata, append_val_into_logs
from train.tempargs import *
import train.model as model
from train.model import GCN, PrimalDualModel
import pickle
import numpy as np
from typing import Union, Dict, Any
from collections import OrderedDict  # 关键导入

import torch
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 插入到最前确保优先级
from train.tempargs import *
from train.utils import getfactordata, append_val_into_logs
import pickle
import torch
import torch.nn as nn
from train.utils import *  # if utils.py is in the parent directory
from train.model import *
import os
import matplotlib.pyplot as plt
import time
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')
from torch.utils.data import  TensorDataset
import train.model as model
from train.main import *
NumK = args.NumK
equal_flag = args.equal_flag
device = args.device()
Bounds = args.Bounds
rate = args.rate
bandwidth = args.bandwidth
numofbyte = args.numofbyte

def testdataprocess():
    Xtest, Ytest , cinfotest = {} , {} , {}
    for PDB in args.PDBs:
        factor_test_data = test_data_path + f'te_factor=0.5_NumK={NumK}.h5'
        Xtest[PDB], Ytest[PDB], cinfotest[PDB] = getfactordata(factor_test_data, PDB, NumK, device=device, equal_flag=equal_flag)
    # print(f"Xtest:{Xtest}, cinfotest:{cinfotest}")
    return  Xtest, Ytest , cinfotest


'''def test_delay(model_path, PDBs):
    Xtest, Ytest , cinfotest = testdataprocess()
    print(Xtest)
    # bound = torch.tensor([Bounds[10]])
    model_name_list = ['HARQ', 'HARQ-CC', 'HARQ-IR',]
    for model_name in model_name_list:
        testlog = {}
        testlog2 = {}
        print(f"model_name:{model_name}")
        model_save_path = model_path +f'\\{model_name}-NumK={NumK}-PDB={args.PDB}_model_pd_satisfy.pt'
        if os.path.exists(model_save_path):
            print(f"model_save_path:{model_save_path}")
            model_pd = torch.load(model_save_path)

            for PDB in PDBs:
                print(PDB, Xtest[PDB][:,[0,-1]])
                model_pd.eval()
                with torch.no_grad():
                    pt = model_pd(Hx_dir={"Hx": Xtest[PDB], 'edge_index': cinfotest[PDB]['edge_index']},
                                      bounds=torch.tensor([Bounds[PDB]]).to(device),
                                      rate=rate,
                                      numofbyte=numofbyte,
                                      bandwidth=bandwidth)
                append_val_into_logs(testlog, PDB, {'delay':model_pd.l_p, 'pout':model_pd.pout[:, -1], 'pt':pt})
                with torch.no_grad():
                    x = Xtest[PDB].clone()
                    x[:, NumK] = PDB*2/NumK
                    x[:, -1] = PDB*2
                    pt = model_pd(Hx_dir={"Hx": x, 'edge_index': cinfotest[PDB]['edge_index']},
                                      bounds=torch.tensor([Bounds[PDB]]).to(device),
                                      rate=rate,
                                      numofbyte=numofbyte,
                                      bandwidth=bandwidth)
                append_val_into_logs(testlog2, PDB*2, {'delay':model_pd.l_p, 'pout':model_pd.pout[:, -1], 'pt':pt})
        else:
            for PDB in PDBs:
                append_val_into_logs(testlog, PDB, {'delay':None, 'pout':None, 'pt':None})

        with open(log_save_path + f'{model_name}_{NumK}_power.pk', 'wb') as f:
            print(f"testlog:{testlog}")
            print(f"testlog2:{testlog2}")
            pickle.dump(testlog, f)'''

def test_auto_pdb_delay(model_path):
    """测试不同 factor、PDB 和模型组合的性能"""
    results = OrderedDict()  # 使用 OrderedDict 确保顺序一致性
    
    for factor in args.factors:
        results[factor] = OrderedDict()
        factor_test_data = test_data_path + f'te_factor={factor}_NumK={NumK}.h5'
        
        for pdb in args.PDBs:
            results[factor][pdb] = OrderedDict()
            
            try:
                # 加载测试数据
                Xtest, Ytest, cinfotest = getfactordata(factor_test_data, pdb, NumK, 
                                                       device=device, equal_flag=equal_flag)
                
                for model_name in model_name_list:  # 使用全局定义的 model_name_list
                    model_save_path = os.path.join(model_path, 
                        f'{model_name}-NumK={NumK}-factor={factor}-PDB={pdb}_model.pt')
                    
                    if os.path.exists(model_save_path):
                        print(f"Testing {model_name} with factor={factor}, PDB={pdb}")
                        try:
                            model_pd = torch.load(model_save_path, weights_only=False)
                            model_pd.eval()
                            
                            with torch.no_grad():
                                pt = model_pd(Hx_dir={"Hx": Xtest, 'edge_index': cinfotest['edge_index']},
                                            bounds=torch.tensor([Bounds[pdb]]).to(device),
                                            rate=rate,
                                            numofbyte=numofbyte,
                                            bandwidth=bandwidth)
                                
                                results[factor][pdb][model_name] = {
                                    'delay': float(model_pd.l_p.item()),
                                    'pout': float(model_pd.pout[:, -1].item()),
                                    'pt': float(pt.mean().item())
                                }
                        except Exception as e:
                            print(f"Error testing model {model_name} for factor={factor}, PDB={pdb}: {str(e)}")
                            results[factor][pdb][model_name] = {
                                'delay': None, 'pout': None, 'pt': None, 'error': str(e)
                            }
                    else:
                        print(f"Model not found: {model_save_path}")
                        results[factor][pdb][model_name] = {
                            'delay': None, 'pout': None, 'pt': None, 'error': 'Model file not found'
                        }
                        
            except Exception as e:
                print(f"Error processing factor={factor}, PDB={pdb}: {str(e)}")
                results[factor][pdb] = {'error': str(e)}

    # 保存完整测试结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(log_save_path, f'complete_test_results_NumK={NumK}_{timestamp}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 保存一个更易读的 CSV 版本
    csv_data = []
    for factor in results:
        for pdb in results[factor]:
            for model_name in results[factor][pdb]:
                if isinstance(results[factor][pdb][model_name], dict):
                    row = {
                        'Factor': factor,
                        'PDB': pdb,
                        'Model': model_name,
                        **results[factor][pdb][model_name]
                    }
                    csv_data.append(row)
    
    csv_path = os.path.join(log_save_path, f'test_results_NumK={NumK}_{timestamp}.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Results saved to:\n{save_path}\n{csv_path}")

    return results

def draw_results(results, save_path):
    """绘制多维度的结果可视化图表"""
    metrics = ['delay', 'pout', 'pt']
    for metric in metrics:
        plt.figure(figsize=(15, 10))
        
        for model_name in model_name_list:
            delays = []
            pdbs = []
            factors = []
            
            for factor in results:
                for pdb in results[factor]:
                    if model_name in results[factor][pdb]:
                        result = results[factor][pdb][model_name]
                        if isinstance(result, dict) and result.get(metric) is not None:
                            delays.append(result[metric])
                            pdbs.append(pdb)
                            factors.append(factor)
            
            if delays:  # 只在有数据时绘图
                plt.scatter(pdbs, delays, label=f'{model_name}', alpha=0.6)
        
        plt.xlabel('PDB (dB)')
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'{metric.capitalize()} vs PDB for Different Models')
        plt.legend()
        plt.grid(True)
        
        save_file = os.path.join(save_path, f'{metric}_comparison_NumK={NumK}.png')
        plt.savefig(save_file)
        plt.close()

if __name__=='__main__':
    try:
        results = test_auto_pdb_delay(model_save_path)
        draw_results(results, graph_path)
        print("Testing and visualization completed successfully.")
    except Exception as e:
        print(f"Error during testing: {str(e)}")