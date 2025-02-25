import sys
import os

# 获取项目根目录（假设脚本在 test/ 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 插入到最前确保优先级

# 确保 utils 模块所在的路径在 sys.path 中
utils_path = os.path.join(project_root, 'train')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

import torch
from train.utils import getdata, getfactordata, append_val_into_logs
from train.tempargs import *
import train.model as model
from train.model import GCN, PrimalDualModel
import pickle
import numpy as np
from typing import Union, Dict, Any
from collections import OrderedDict  # 关键导入

seed = args.seed
NumK = args.NumK
dropout = args.dropout
PDB = args.PDB
Bounds = args.Bounds
rate = args.rate
bandwidth = args.bandwidth
numofbyte = args.numofbyte
equal_flag = args.equal_flag
device = args.device()
in_size = args.in_size
out_size = args.out_size
inter = args.inter

def testdataprocess():
    Xtest, Ytest , cinfotest = {} , {} , {}
    for factor in args.factors:
        factor_test_data = test_data_path + f'te_factor={factor}_NumK={NumK}.h5'
        Xtest[factor], Ytest[factor], cinfotest[factor] = getfactordata(factor_test_data, PDB, NumK, device=device, equal_flag=equal_flag)
    # print(f"Xtest:{Xtest}, cinfotest:{cinfotest}")
    return  Xtest, Ytest , cinfotest

def test_delay(model_path):
    Xtest, Ytest , cinfotest = testdataprocess()
    print(Xtest)
    bound = torch.tensor([Bounds[PDB]]).to(device)
    model_name_list = [ 'HARQ', 'HARQ-CC', 'HARQ-IR',]

    for model_name in model_name_list:
        testlog = {}
        print(f"model_name:{model_name}")

        # gcnmodel = GCN(in_size=in_size,
        #                out_size=out_size,
        #                **inter[0],
        #                dropout=dropout,
        #                NumK=NumK,
        #                edge_index=None).to(device)
        #
        # model_pd = PrimalDualModel(model_name,
        #                            model=gcnmodel,
        #                            Numk=NumK,
        #                            constraints=['pout', 'power'],
        #                            device=device)
        #save_path = model_path +f'{model_name}-NumK={NumK}-PDB={PDB}_model_pd_satisfy.pt'
        save_path = model_path +f'{model_name}-NumK={NumK}_avg.pt'
        print(f"model_save_path:{save_path}")
        #model_pd = torch.load(save_path)
         # 安全加载模型（需 PyTorch >= 2.6）
        model_pd = torch.load(save_path, weights_only=False)  # 显式启用安全模式
        for factor in args.factors:
            with torch.no_grad():
                model_pd.eval()
                pt = model_pd(Hx_dir={"Hx": Xtest[factor], 'edge_index': cinfotest[factor]['edge_index']},
                                  bounds=bound,
                                  rate=rate,
                                  numofbyte=numofbyte,
                                  bandwidth=bandwidth)
                if model_pd.pout[:, -1]<= bound:
                    append_val_into_logs(testlog, factor, {'delay':model_pd.l_p, 'pout':model_pd.pout[:, -1], 'pt':pt})
                else:
                    append_val_into_logs(testlog, factor, {'delay':None, 'pout':None, 'pt':None})

        #with open(log_save_path + f'{model_name}_{PDB}_{NumK}_factor.pk', 'wb') as f:
        with open(log_save_path + f'{model_name}_avg_{NumK}_factor.pk', 'wb') as f:
            print(f"testlog:{testlog}")
            pickle.dump(testlog, f)
if __name__=='__main__':
    test_delay(model_save_path)