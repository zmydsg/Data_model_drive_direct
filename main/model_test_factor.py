import torch
from utils import getdata, getfactordata, append_val_into_logs
from tempargs import args
from model import GCN,PrimalDualModel
import pickle

seed = args.seed
NumK = args.NumK
dropout = args.dropout
PDB = args.PDB
Bounds = args.Bounds
rate = args.rate
bandwidth = args.bandwidth
numofbyte = args.numofbyte
equal_flag = args.equal_flag
device = args.device
in_size = args.in_size
out_size = args.out_size
inter = args.inter
test_data_path = '.\\dataset\\'
log_save_path = '.\\testlog\\'

def testdataprocess():
    Xtest, Ytest , cinfotest = {} , {} , {}
    for factor in [0, 0.1, 0.2, 0.3, 0.4 ,0.5 , 0.6 ,0.7, 0.8, 0.9 ,0.92, 0.94, 0.96, 0.98]:
        factor_test_data = test_data_path + f'te_factor={factor}_NumK={NumK}.h5'
        Xtest[factor], Ytest[factor], cinfotest[factor] = getfactordata(factor_test_data, PDB, NumK, device=device, equal_flag=equal_flag)
    # print(f"Xtest:{Xtest}, cinfotest:{cinfotest}")
    return  Xtest, Ytest , cinfotest

def test_delay(model_path):
    Xtest, Ytest , cinfotest = testdataprocess()
    print(Xtest)
    bound = torch.tensor([Bounds[PDB]])
    model_name_list = [ 'HARQ-CC',]
    testlog = {}
    for model_name in model_name_list:
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
        model_save_path = model_path +f'\\{model_name}-NumK={NumK}-PDB={PDB}_model_pd_satisfy.pt'
        print(f"model_save_path:{model_save_path}")
        model_pd = torch.load(model_save_path)

        for factor in [ 0, 0.1, 0.2, 0.3, 0.4 ,0.5 , 0.6 ,0.7, 0.8, 0.9 ,0.92, 0.94, 0.96, 0.98]:
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

        with open(log_save_path + f'{model_name}_{PDB}_{NumK}_factor.pk', 'wb') as f:
            print(f"testlog:{testlog}")
            pickle.dump(testlog, f)
if __name__=='__main__':
    model_path = ".\\model"
    test_delay(model_path)