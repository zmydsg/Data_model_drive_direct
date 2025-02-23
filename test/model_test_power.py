import torch
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


def test_delay(model_path, PDBs):
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
            pickle.dump(testlog, f)

def test_auto_pdb_delay(model_path, PDBs):
    Xtest, Ytest , cinfotest = testdataprocess()
    print(Xtest)
    # bound = torch.tensor([Bounds[10]])
    model_name_list = ['HARQ',"HARQ-CC","HARQ-IR"]
    for model_name in model_name_list:
        testlog = {}
        print(f"model_name:{model_name}")
        model_save_path = model_path +f'\\{model_name}-NumK={NumK}_avg.pt'
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
        else:
            for PDB in PDBs:
                append_val_into_logs(testlog, PDB, {'delay':None, 'pout':None, 'pt':None})

        with open(log_save_path + f'{model_name}_{NumK}_power_avg.pk', 'wb') as f:
            print(f"testlog:{testlog}")
            pickle.dump(testlog, f)

if __name__=='__main__':
    test_auto_pdb_delay(model_save_path, args.PDBs) #有avg

    #test_delay(model_save_path, args.PDBs)     #无avg