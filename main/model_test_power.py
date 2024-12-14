import torch
from utils import getdata, getfactordata, setup_seed , append_val_into_logs
from tempargs import args
from model import GCN,PrimalDualModel
import pickle
import os

NumK = args.NumK
equal_flag = args.equal_flag
device = args.device
Bounds = args.Bounds
rate = args.rate
bandwidth = args.bandwidth
numofbyte = args.numofbyte
test_data_path = '.\\dataset\\'
log_save_path = '.\\testlog\\'

def testdataprocess():
    Xtest, Ytest , cinfotest = {} , {} , {}
    for PDB in [10, 12, 14, 16, 18, 20, 22, 24]:
        factor_test_data = test_data_path + f'te_factor=0.5_NumK=3.h5'
        Xtest[PDB], Ytest[PDB], cinfotest[PDB] = getfactordata(factor_test_data, PDB, NumK, device=device, equal_flag=equal_flag)
    # print(f"Xtest:{Xtest}, cinfotest:{cinfotest}")
    return  Xtest, Ytest , cinfotest


def test_delay(model_path, PDBs):
    Xtest, Ytest , cinfotest = testdataprocess()
    print(Xtest)
    bound = torch.tensor([Bounds[10]])
    model_name_list = [ 'HARQ-IR',]
    testlog = {}
    for model_name in model_name_list:
        print(f"model_name:{model_name}")

        for PDB in PDBs:
            model_save_path = model_path +f'\\{model_name}-NumK={NumK}-PDB={PDB}_model_pd_satisfy.pt'
            if os.path.exists(model_save_path):
                print(f"model_save_path:{model_save_path}")
                model_pd = torch.load(model_save_path)

                model_pd.eval()
                with torch.no_grad():
                    pt = model_pd(Hx_dir={"Hx": Xtest[PDB], 'edge_index': cinfotest[PDB]['edge_index']},
                                      bounds=bound,
                                      rate=rate,
                                      numofbyte=numofbyte,
                                      bandwidth=bandwidth)
                append_val_into_logs(testlog, PDB, {'delay':model_pd.l_p, 'pout':model_pd.pout[:, -1], 'pt':pt})
            else:
                append_val_into_logs(testlog, PDB, {'delay':None, 'pout':None, 'pt':None})

        with open(log_save_path + f'{model_name}_{NumK}_power.pk', 'wb') as f:
            print(f"testlog:{testlog}")
            pickle.dump(testlog, f)

if __name__=='__main__':
    model_path = ".\\model"
    PDBs = [10, 12, 14, 16, 18, 20, 22, 24]
    test_delay(model_path, PDBs)