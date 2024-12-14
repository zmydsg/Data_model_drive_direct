import numpy as np
import h5py
from prettytable import PrettyTable
from torch_geometric.utils import dense_to_sparse
import os
import torch
import math
from torch.utils.data import TensorDataset,DataLoader
import sys
import random
import builtins as __builtin__
from datetime import datetime

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#生成文件路口
def generateFilePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

#初始化随机种子
def setup_seed(seed):
    """
    stable the random
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(dpath, **kwargs):
    """
    读取数据
    """
    X = []
    pin = np.array(10**(kwargs['PDB']/10))

    with h5py.File(dpath,'r') as handle:
        
        hs = handle['input']['channel']
        datanum1,datanum2 = hs.shape
        reshapesize = int(np.power(datanum2-1,0.5))

        for index in range(datanum1):

            factor = hs[index,-1]

            edge_index,h = dense_to_sparse(torch.from_numpy(hs[index,0:-1].reshape(reshapesize,reshapesize).astype(float)))

            x1 = np.hstack([h,factor,pin]).reshape(-1,1).T
            X.append(x1)
        cinfo={
            'edge_index':edge_index
        }
    X = np.concatenate((X))
    X = X[~np.any(np.isnan(X),-1)]
    
    y = np.full([X.shape[0], reshapesize], np.nan)

    return X, y, cinfo


def getdata(data_path, pdb, NumK, **kwargs):
    """
    读取产生的数据，并将初始功率拼接上去
    """
    equal_flag = kwargs['equal_flag']
    attach_init_power = lambda x, NumK: np.tile(np.random.dirichlet(np.ones(NumK),size=1),(int(x.shape[0]),1))*x[0,-1]
    attach_equal_power = lambda x, NumK: np.tile(np.ones((1,NumK)),(int(x.shape[0]),1))*(x[0,-1]/NumK)
    func_to_tensor = lambda x, pt, device:  torch.from_numpy(np.hstack([pt,x])).float().to(device)
    dict_to_tensor = lambda x,device:  {k:v.to(device) if isinstance(v ,torch.Tensor) else v for k ,v in x.items()}



    x ,y ,cinfo = {} , {} , {}
    # locate = kwargs['locate']
    #
    # phase = 'tr'
    # project_path = os.getcwd()
    # data_path = os.getcwd()+'\\dataset\\'
    #
    # path = data_path + f"{phase}_direct_{NumK}.h5"
    # print(f"\nload data path from:{path}")

    listtoprocess = ['tr','val']

    for phase in listtoprocess:
        dpath = data_path+f'{phase}_inverse_direct_NumK={NumK}.h5'
        x[phase], y[phase], cinfo[phase] = load_data(dpath, PDB=pdb)

        if equal_flag:
            pt = attach_equal_power(x[phase],NumK)
        else:
            pt = attach_init_power(x[phase],NumK)

        x[phase] = func_to_tensor(x[phase], pt, kwargs['device'])
        y[phase] = x[phase][:,:NumK]
        cinfo[phase] = dict_to_tensor(cinfo[phase],kwargs['device'])

    return x , y ,cinfo

def getfactordata(datapath, pdb, NumK, **kwargs):
    """
    读取产生的数据，并将初始功率拼接上去
    """
    equal_flag = kwargs['equal_flag']
    attach_init_power = lambda x, NumK: np.tile(np.random.dirichlet(np.ones(NumK),size=1),(int(x.shape[0]),1))*x[0,-1]
    attach_equal_power = lambda x, NumK: np.tile(np.ones((1,NumK)),(int(x.shape[0]),1))*(x[0,-1]/NumK)
    func_to_tensor = lambda x, pt, device:  torch.from_numpy(np.hstack([pt,x])).float().to(device)
    dict_to_tensor = lambda x,device:  {k:v.to(device) if isinstance(v ,torch.Tensor) else v for k ,v in x.items()}


    x ,y ,cinfo = {} , {} , {}
    # locate = kwargs['locate']
    #
    # phase = 'tr'
    # project_path = os.getcwd()
    # data_path = os.getcwd()+'\\dataset\\'
    #
    # path = data_path + f"{phase}_direct_{NumK}.h5"
    # print(f"\nload data path from:{path}")

    x, y, cinfo = load_data(datapath, PDB=pdb)

    if equal_flag:
        pt = attach_equal_power(x,NumK)
    else:
        pt = attach_init_power(x,NumK)

    x = func_to_tensor(x, pt, kwargs['device'])
    y = x[:,:NumK]
    cinfo = dict_to_tensor(cinfo,kwargs['device'])

    return x , y ,cinfo

def edge_index_batch(edgeindex, numhx, NumK, dev):
    #有向图
    listshift = torch.vstack([torch.arange(numhx) * NumK, ] * (int(NumK*(NumK+1)/2))).T.reshape(-1).repeat(1, 2).view(2, -1).long().to(dev)
    #无向图
    # listshift = torch.vstack([torch.arange(numhx) * NumK, ] *NumK**2).T.reshape(-1).repeat(1,2).view(2, -1).long().to(dev)
    edgeindex_batch = edgeindex.repeat(1, numhx) + listshift
    return edgeindex_batch

def get_model_para(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def funct_cal_HARQ_IR(result, pt, factor, NumK, rate_2):
    """
    HARQ_IR 中断概率计算
    """
    f = lambda x: f(x-1)*x if x>=2 else 1
    f1 = lambda x, m : x**(2*m)  # 2m次方

    a2 = math.log(rate_2, math.e)

    for i in range(1,NumK+1):  # 1， 2， 3，..., K
        # print(f"i:{i}")

        # A : GK(x)
        a0 =0
        for j in range(i):     # 0, 1, 2, ..., K-1
            a1 = f(i-j-1)
            a0 += ((-1)**j)*(a2**(i-j-1))/a1
        a3 = (-1)**i+rate_2*a0

        # B : L(factor ,K)
        a4 = 1
        a5 = 1
        for k in range(1, i+1):
            # print(f"\tk:{k}")
            # a6 = factor**(2*k)
            a6 = f1(factor, k)
            a4 += a6/(1-a6)
            a5 *= (1-a6)

        a7 = 1/(a4*a5)

        # C
        a8 = 1
        for n in range(i):
            # print(f"\t\tn:{n}")
            a8 *=pt[n]

        # A*B*C
        # print("a3*a7:{},a8:{}".format(a3*a7,a8))
        result[i-1]=a3*a7/a8

    return result

def funct_cal_HARQ_CC(result, pt, factor, NumK, rate_2):
    """
    HARQ_CC 中断概率计算
    """
    f = lambda x: f(x-1)*x if x>=2 else 1
    f1 = lambda x, m : x**(2*m)

    for i in range(1,NumK+1):
        a0 = (rate_2-1) ** i
        a1 = f(i)
        # L(factor ,K)
        a4 = 1
        a5 = 1
        for k in range(1,i+1):
            a6 = f1(factor, k)
            a4 +=a6/(1-a6)
            a5 *=(1-a6)
        a7 = 1/(a4*a5)

        a8 = 1
        for n in range(i):
            # print(f"\t\tn:{n}")
            a8 *=pt[n]

        result[i-1]=a7*a0/(a1*a8)

    return result

def funct_cal_HARQ(result, pt, factor, NumK, rate_2):
    """
    1型 HARQ 中断概率计算
    """
    f1 = lambda x, m : x**(2*m)

    for i in range(1,NumK+1):

        a0 = (rate_2 - 1) ** i

        # L(factor ,K)
        a4 = 1
        a5 = 1
        for k in range(1,i+1):
            a6 = f1(factor, k)
            a4 += a6/(1-a6)
            a5 *= (1-a6)
        a7 = 1/(a4*a5)

        a8 = 1
        for n in range(i):
            # print(f"\t\tn:{n}")
            a8 *=pt[n]

        result[i-1]=a7*a0/a8
    return result

def get_utility_func(m):
    func_dict = {
        'HARQ-IR': funct_cal_HARQ_IR,
        'HARQ-CC': funct_cal_HARQ_CC,
        'HARQ': funct_cal_HARQ,
    }
    return func_dict[m]

def Probabilty_outage(outage_cal_func, pt, factors, rate, NumK, NumE, flag = None):

    rate_2 = 2**rate
    if not flag:
        result = torch.zeros_like(pt)
    else:
        result = np.ones_like(pt)

    for i in range(NumE):
        factor = factors[i]
        outage_cal_func(result[i,:], pt[i,:], factor, NumK, rate_2)
    return result
    # return torch.clamp(result, min= 1e-12, max=torch.tensor(1))


def compute_p(pout, pt):
    middle1 = pt[:,1:]
    middle2 = pout[:,0:-1]
    middle3 = (middle1*middle2).sum(dim=1,keepdim=True)
    pt1 = pt[:,0].view(-1,1)
    return pt1+middle3


def through_output(pout , rate ,NumK, flag =None):
    ### numpy 跟pyotrch中对于数组形状操作的函数是不一样的
    if not flag:
        up = rate * (1-pout[:,-1]).view(-1,1)
        down = 1+torch.sum(pout[:,:NumK-1],dim=1).view(-1,1)
    else:
        up = rate * (1 - pout[:, -1]).reshape(-1, 1)
        down = 1 + np.sum(pout[:, :NumK - 1], axis=1).reshape(-1, 1)

    return up/down

def delay_compute(throughput, numofbyte, bandwith):
    mid = throughput* bandwith
    res = numofbyte/ mid
    return res

def append_val_into_logs(logs,key,value):
    if key in logs:
        logs[key].append(value)
    else:
        logs[key] = [value]
    # logs[key] = value
    return logs

def change_val_into_logs(logs,key,value):
    logs[key] = value
    return logs

def val_test_logs(valLogs, model, valdata,modelsavepath = None,constrain = None, **other):
    print("eval()")
    model.eval()
    with torch.no_grad():
        pt = model(**valdata)
        l_P = model.l_p.detach().item()
        
        print(model.delay.numpy().reshape(10,20).tolist())
    epoch = other["epochnum"]

    pout_flag, power_flag = False , False
    for kc, Ef in model.ef.items():
        Efvalus = Ef.cpu().detach().numpy()
        print(f"kc :{kc}\t Efvalus:{Efvalus}")
        
        if (constrain is not None) and (kc =='pout') and (Efvalus.mean()<= 0):
            pout_flag = True
        else:
            pass
        if (constrain is not None) and (kc =='power') and (Efvalus.mean()<= 0):
            power_flag = True
        else:
            pass

    print(f"mean delay:{l_P}")
    print(f"pout_flag:{pout_flag}\t power_flag:{power_flag} \tl_P>= 0.05:{l_P>= 0.05}")

    if power_flag and pout_flag and l_P>= 0.05:
        append_val_into_logs(valLogs , 'satisfy_l_p' , l_P)
        print(f"logs['satisfy_l_p'][-1]:{valLogs['satisfy_l_p'][-1]}\n")
        print(f"logs['satisfy_l_p']:{valLogs['satisfy_l_p']}\n")
        print(f"np.all(l_P <= np.array(valLogs['satisfy_l_p'])):{np.all(l_P <= np.array(valLogs['satisfy_l_p']))}")
        if np.all(l_P <= np.array(valLogs['satisfy_l_p'])) and modelsavepath:
            print(f"save model in epoch:{epoch}" )
            torch.save(model , modelsavepath+'_model_pd_satisfy.pt')
    else:
        pass

    return valLogs





            
    



