import numpy as np
import torch
from train.utils import *
from train.model import *
import os
import matplotlib.pyplot as plt
import time
import warnings
from tempargs import *
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')
# from torch.utils.data import  TensorDataset

# 初始化参数
epochs = args.epochs
seed = args.seed
NumK = args.NumK
dropout = args.dropout
learning_rate = args.learning_rate
learning_rate2 = args.learning_rate2
learning_rate3 = args.learning_rate3

Bounds = args.Bounds
rate = args.rate
bandwidth = args.bandwidth
numofbyte = args.numofbyte
equal_flag = args.equal_flag
device = args.device()
in_size = args.in_size
out_size = args.out_size
inter = args.inter#[
        #{'inter_size': [4, 8, 1],
         #   'inter_activation': ['elu', 'relu',  'linear'],
         #}]
# model_name = args.model_name
# PDB = args.PDB
# dual element update stepsize
update_Step = {"pout": learning_rate2, "power": learning_rate3, 'init': learning_rate}

# get data_path
# project_path = os.getcwd(),不用绝对路径，用相对路径
train_data_name = data_path + f'tr_inverse_direct_Numk={NumK}.h5'
test_data_name = data_path + f'te_inverse_direct_NumK={NumK}.h5'

# 判断控制台输出存取的目录是否存在
generateFilePath(print_save_path)
# 判断保存对应模型图片的是否存在
generateFilePath(photo_save_path)
# 判断保存模型数据的路径是否存在
generateFilePath(model_save_path)
generateFilePath(tensorlog_save_path)

# tensorboard 记录吞吐量变化
writer = SummaryWriter(tensorlog_save_path)

model_name_list = ["HARQ"]#,"HARQ-CC","HARQ-IR"]


def train(eval=False):
    if eval and not os.path.exists(val_path): os.mkdir(val_path)
    for model_name in model_name_list:
        modelsavepathm = model_save_path + f'{model_name}-NumN={NumN}-NumK={NumK}_model_pd_m.pt'
        valpath = val_path + f'{model_name}-NumN={NumN}-NumK={NumK}'
        X = {'tr': [], 'val': []}
        Y = {'tr':[],'val':[]}
        model_pds = []
        pms=[]
        if eval:
            valLogs= {}
            valdata = {
                'Hx_dirs':0, 'edge_index_': 0, 'delays': delays, "pms":pms, "avg":AVG,
                'bounds':[],'rate':rate,'numofbyte':numofbyte,'bandwidth':bandwidth}

        for PDB,bound in zip(PDBS, bounds):
            #初始化固定随机种子
            setup_seed(seed)
            #模型保存路径
            modelsavepath = model_save_path + f'{model_name}-NumK={NumK}-PDB={PDB}_model_pd_satisfy.pt'

            bound = torch.log10(torch.tensor([bound])).to(device)
            pms.append(PDB)#10**(PDB/10))
            if eval:
                valdata['bounds'].append(bound)
            # init model and paras
            model_pds.append(torch.load(modelsavepath))
            model_pds[-1].eval()
            model_pds[-1].model.eval()
            model_pds[-1].model.model.eval()

        # data-process
        x, y, cinfo = getdata(data_path,PDB, NumK, device=device, equal_flag=equal_flag, eval=eval)
        X['tr']=x['tr']
        if eval:
            X['val']=x['val']
        print(model_name, f"\nbounds:{bounds}\tdb:{PDB}\tPt_max:{x['tr'][0,-1]}")

        gcnmodel = GCN(in_size=in_size,
                       out_size=out_size,
                       **inter[0],
                       dropout=dropout,
                       NumK=NumN,
                       edge_index=None).to(device)

        model_pdm = MultiModel(
            gcn_model=gcnmodel,
            pd_models=model_pds,
            NumN=NumN, NumK=NumK,
            device=device).to(device)

        optimizer = torch.optim.Adam(gcnmodel.parameters(), lr=learning_rate)
        # 动态学习率，调整learning rate大小
        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100*20, gamma=0.4)
        l_p_list=[]
        delay_list=[]
        sinr_list=[]
        ltat_list=[]
        batch_size = 16
        data_sum,val_sum = len(X['tr']),len(X['val'])
        # x_ = np.zeros((16, NumN, NumK+NumK*(NumK+1)//2+2), dtype=torch.float32)
        x = torch.empty((batch_size, NumN, NumK+NumK*(NumK+1)//2+2), dtype=torch.float32, requires_grad=False)
        if eval:
            valdata['Hx_dirs'] =x.clone()
        try:
            for epoch in range(epochs):
                print("=" * 100)
                print(f"\nepoch:{epoch}")
                for i in range(data_sum//(batch_size*NumN)):
                    x[:] = X['tr'][np.random.randint(0, data_sum, (batch_size,NumN), dtype=np.int16)]
                    for j in range(NumN):
                        x[:, j, -1] = PDBS[j]
                    model_pdm.forward(Hx_dirs=x,
                                      edge_index_=cinfo['tr']['edge_index'],
                                  bounds=bounds,
                              delays=delays,
                                      pms=pms,
                                      avg=AVG,
                                  rate=rate,
                                  numofbyte=numofbyte,
                                  bandwidth=bandwidth)

                    optimizer.zero_grad()
                    model_pdm.l_p.backward(retain_graph=True)    #################################

                    torch.nn.utils.clip_grad_norm_(model_pdm.parameters(), 3)
                    optimizer.step()
                    lr_sch.step()

                    # 存取训练中变化
                    if not i%20:
                        delay = model_pdm.delay.item()
                        sinr = model_pdm.l_p.item()
                        ltat = model_pdm.l_p.item()
                        lp = model_pdm.l_p.item()
                        print("delay:", delay,
                              " SINR_MAX:", sinr,
                              " LTAT_MAX:", ltat,
                              )
                        delay_list.append(delay)
                        l_p_list.append(lp)
                        sinr_list.append(sinr)
                        ltat_list.append(ltat)
                        writer.add_scalar(tag=f"{model_name}-l_p", scalar_value=lp, global_step=epoch*20+i)
                        writer.add_scalar(tag=f"{model_name}-delay", scalar_value=delay,global_step=epoch * 20 + i)
                        writer.add_scalar(tag=f"{model_name}-LTAT_MAX", scalar_value=ltat,global_step=epoch * 20 + i)
                        writer.add_scalar(tag=f"{model_name}-SINR_MAX", scalar_value=sinr,global_step=epoch * 20 + i)
                if eval:
                    # 验证部分
                    model_pdm.eval()
                    valdata['Hx_dirs'][:] = X['val'][np.random.randint(0, val_sum, (batch_size, NumN), dtype=np.int16)]
                    valdata['edge_index_'] = cinfo['val']['edge_index']
                    with torch.no_grad():
                        pt_vals = model_pdm(**valdata)
                        # 计算验证集上的时延和中断概率
                        delay_val = model_pdm.delay.item()
                        delay_vals = [model_pd.l_p.item() for model_pd in model_pds]
                        pout_vals = [model_pd.pout.mean().item() for model_pd in model_pds]
                        pt_vals = [pt.mean().item() for pt in pt_vals]
                    model_pdm.train()

                    valLogs[epoch] = {'delay_val': delay_val, 'delay_vals': delay_vals, 'pt_val': pt_vals, 'pout_val': pout_vals}
                    print(f'Delay: {delay_val}, Pout: {pout_vals}')
                    writer.add_scalar(tag=f"{model_name}-val-delay", scalar_value=delay_val, global_step=epoch)
                    for n,v in enumerate(pout_vals):
                        writer.add_scalar(tag=f"{model_name}-val-pout{n}", scalar_value=v, global_step=epoch)

        except KeyboardInterrupt:
            i = input('Save?').lower()
            if i == 't':
                torch.save(model_pdm, modelsavepathm)
                break
            elif i == 'f':
                break
        torch.save(model_pdm, modelsavepathm)

        if eval:
            # 保存val验证集的log文件
            import pickle
            with open(valpath+'_power_vallog_m.pk', 'wb') as f:
                pickle.dump(valLogs, f)

        # from findFuncAnswer import equalAllocation
        # equalAllocation(1000, factor, NumK, rate, bound, func=get_utility_func(model_name))
        #matplot draw graph
        x = np.linspace(start=0,stop=len(delay_list),num=len(delay_list))
        plt.figure(figsize=(16, 8))
        plt.subplot(1,2,1)
        plt.title(f"{model_name} epoch---delay", color='b')
        plt.xlabel("epoch")
        plt.ylabel("delay")
        plt.plot(x, delay_list)
        if any(0>delay > 0.5 for delay in delay_list):
            plt.ylim(0, 0.5)

        plt.subplot(1,2,2)
        plt.title("epoch---square", color='b')
        plt.xlabel("epoch")
        plt.ylabel("square")
        plt.plot(x, l_p_list)
        if any(lagr > 10 for lagr in l_p_list):
            plt.ylim(None, 10)

        plt.savefig(photo_save_path+f'\\{model_name}-NumN={NumN}-NumK={NumK}-equal_flag={equal_flag}_m.jpg')
        plt.show()

        plt.figure(figsize=(16, 8))
        plt.subplot(1,2,1)
        plt.title(f"{model_name} epoch---SINR_MAX", color='b')
        plt.xlabel("epoch")
        plt.ylabel("SINR")
        plt.plot(x, sinr_list)
        plt.subplot(1,2,2)
        plt.title("epoch---LTAT_MAX", color='b')
        plt.xlabel("epoch")
        plt.ylabel("LTAT")
        plt.plot(x, ltat_list)
        plt.savefig(photo_save_path+f'\\{model_name}-NumN={NumN}-NumK={NumK}-equal={equal_flag}_m1.jpg')
        plt.show()

    #关闭tensorboard写入
    writer.close()


if __name__ =="__main__":
    factor=0.50
    factors_val=args.factors
    NumN=4
    AVG=1.5
    delays = [0.1]*NumN
    PDBS  = [11,15,19,23]
    bounds = [args.Bounds[PDB] for PDB in PDBS]
    train(True)
