import numpy
import torch
from tempargs import  args
import os
import numpy as np
import h5py
from utils import getdata,load_data

equal_flag = args.equal_flag
NumK = args.NumK
trainSeed= args.train_seed
valSeed = args.val_seed
device = args.device

# get data_path
# project_path = os.getcwd()
project_path = ".\\"
data_path = project_path + 'dataset\\'
train_data_name = data_path + f'tr_inverse_direct_Numk={NumK}.h5'
val_data_name = data_path +f'val_inverse_direct_Numk={NumK}.h5'


#非对角元素为factor^(i+j)
def generate1example(factor,K,x):
    """
    # 产生无向图模型
    """
    func1 = lambda a,index: np.power(a,2*index)

    for i in range(1,K+1):
        #a1 = func1(factor,i)

        for j in range(1,K+1):
            #a2 = func1(factor,j)

            if i != j:
                #result =1-a1+a1*(1-a2)+8*a1*a2
                result = factor**(i+j)
                x[i-1,j-1]=result
                #print("i:{},j:{},result:{}\t".format(i,j,result))
            else:
                x[i - 1, j - 1] = 1
    return x.reshape(1,-1)


#上三角矩阵
def generate1DirectExample(factor,K,x):
    """
    #产生有向图模型
    """
    for i in range(1,K+1):
        for j in range(1,K+1):
            if j == i:
                x[i-1, j-1] = 1
            elif j > i:
                result = factor**(i+j)
                x[i - 1, j - 1] = result

    return x.reshape(1, -1)

#下三角矩阵
def generate1InverseDirectExample(factor,K,x):
    """
    #产生有向图模型
    """
    for i in range(1,K+1):
        for j in range(1,K+1):
            if j == i:
                x[i-1, j-1] = 1
            elif j < i:
                result = factor**(i+j)
                x[i - 1, j - 1] = result
    print(x)
    return x.reshape(1, -1)

#根据指定数量 dataNum 和 seed 随机产生 factor，为每个 factor 生成一个(K,K)矩阵，
def makedata(dpath, K, dataNum, seed):
    """
    根据输入的factor
    产生对应的一组数据
    """
    with h5py.File(dpath,'w-')as f:
        g = f.create_group('input')
        dset = g.create_dataset('channel',(dataNum,K**2+1),dtype = np.float32)

        np.random.seed(seed)
        factor_list = np.random.rand(dataNum)
        for i in range(dataNum):
            
            # 产生有向图模型
            factor = factor_list[i]
            x = np.zeros((K, K), dtype=np.float32)
            #通过 generate1DirectExample 将其转化为输入特征，再加入 factor 值组成（K^2+1）维的输入特征行向量，
            x = generate1DirectExample(factor, K, x)#
            #并最终写入到 h5 文件中。
            dset[i, :-1], dset[i, -1] = x, factor

        y = x.reshape(K,K)
        print("X:{},X.shape:{}".format(y,y.shape))
        print("dset:{},dset.shape:{}".format(dset[1],dset[1].shape))
        print(f"generate {dpath} success\n")



#############################

def makedataAccordingfactor(data_path, pdb, NumK, **kwargs):
    """
    读取产生的数据，并将初始功率拼接上去
    """
    equal_flag = kwargs['equal_flag']
    test_flag = kwargs['test_flag']
    attach_init_power = lambda x, NumK: np.tile(np.random.dirichlet(np.ones(NumK),size=1),(int(x.shape(0),1)))*x[0,-1]
    attach_equal_power = lambda x, NumK: np.tile(np.ones((1,NumK)),(int(x.shape[0]),1))*(x[0,-1]/NumK)
    func_to_tensor = lambda x, pt, device:  torch.from_numpy(np.hstack([pt,x])).float().to(device)
    dict_to_tensor = lambda x,device:  {k:v.to(device) if isinstance(v ,torch.Tensor) else v for k ,v in x.items()}

    x ,y ,cinfo = {} , {} , {}

    phase = 'te'
    x[phase], y[phase], cinfo[phase] = load_data(data_path, PDB=pdb)

    if equal_flag:
        pt = attach_equal_power(x[phase],NumK)
    else:
        pt = attach_init_power(x[phase],NumK)

    x[phase] = func_to_tensor(x[phase], pt, kwargs['device'])
    y[phase] = x[phase][:,:NumK]
    cinfo[phase] = dict_to_tensor(cinfo[phase],kwargs['device'])

    return x , y ,cinfo

def makefactordata(dpath, K, factor):
    print(f"dpath:{dpath}")
    with h5py.File(dpath,'w-')as f:
        g = f.create_group('input')
        dset = g.create_dataset('channel',(1,K**2+1),dtype = np.float32)

            # 产生有向图模型
        x = np.zeros((K, K), dtype=np.float32)
        x = generate1DirectExample(factor, K, x)
        dset[0, :-1], dset[0, -1] = x, factor

        y = x.reshape(K,K)
        print("X:{},X.shape:{}".format(y,y.shape))
        print("dset:{},dset.shape:{}".format(dset[0],dset[0].shape))
        print(f"generate {dpath} success\n")
####################################################

# 判断读取数据是否存在
def generateDataSet(dataname, dataNum, seed):
    """
    #### 此部分为生成路径选项
    """
    print(f"dataname:{dataname}")
    if not os.path.exists(dataname):
        print("文件不存在，正在生成")
        makedata(dpath=dataname, K=NumK, dataNum=dataNum, seed= seed)
    else:
        print("Continue")
        pass

if __name__=='__main__':
    
    factor = 0.98
    factor_test_data = data_path + f'te_factor={factor}_NumK={NumK}.h5'
    # makefactordata(factor_test_data, K=NumK, factor=factor)
    # generateDataSet(train_data_name, 50, args.train_seed)
    generateDataSet(val_data_name, 100, args.val_seed)