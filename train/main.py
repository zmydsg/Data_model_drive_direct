import torch
from train.utils import *
from train.model import GCN, PrimalDualModel
import os
import matplotlib.pyplot as plt
import time
import warnings
from torch_geometric.utils import is_undirected
from tempargs import *
import random
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')
from torch.utils.data import  TensorDataset

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
inter = args.inter
# model_name = args.model_name
# PDB = args.PDB
# dual element update stepsize
update_Step = {"pout": learning_rate2, "power": learning_rate3}

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

model_name_list = ["HARQ","HARQ-CC","HARQ-IR"]

PDBS  = [15]

def train():
    for model_name in model_name_list:
        for PDB in PDBS:
            #初始化固定随机种子
            setup_seed(seed)

            # 生成存储控制台输出的文件路径
            logsrecord_name = print_save_path + f'{model_name}-PDB={PDB}-Numk={NumK}-'+time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                            time.localtime()) + '.log'
            
            #模型保存路径
            modelsavepath = model_save_path + f'{model_name}-NumK={NumK}-PDB={PDB}_model_pd_satisfy.pt'
            valpath = val_path + f'{model_name}-NumK={NumK}-PDB={PDB}'
            
            # 记录正常的 print 信息
            sys.stdout = Logger(logsrecord_name)
            # 记录 traceback 异常信息
            sys.stderr = Logger(logsrecord_name)

            bound = torch.log10(torch.tensor([Bounds[PDB]])).to(device)

            # data-process
            X, Y, cinfo = getdata(data_path,PDB, NumK, device=device, equal_flag=equal_flag)

            dataset = TensorDataset(X['tr'] , Y['tr'])
            dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


            # valLogs= {}
            # valdata = {
            #     'Hx_dir':{'Hx':X['val'], 'edge_index': cinfo['val']['edge_index']},
            #     'bounds':bound,'rate':rate,'numofbyte':numofbyte,'bandwidth':bandwidth}
            

            print(f"\nbounds:{bound}\tdb:{PDB}\tPt_max:{X['tr'][0,-1]}")

            start_time = time.time()

            # init model and paras
            gcnmodel = GCN(in_size=in_size,
                           out_size=out_size,
                           **inter[0],
                           dropout=dropout,
                           NumK=NumK,
                           edge_index=None).to(device)

            model_pd = PrimalDualModel(model_name,
                                       model=gcnmodel,
                                       Numk=NumK,
                                       constraints=['pout', 'power'],
                                       device=device).to(device)

            optimizer = torch.optim.Adam(gcnmodel.parameters(), lr=learning_rate)

            # 动态学习率，调整learning rate大小
            lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100*20, gamma=0.4)
            l_p_list = []
            lagr_list = []

            try:
                # training
                for epoch in range(epochs):
                    print("=" * 100)
                    print("=" * 100)
                    print("=" * 100)
                    print(f"\nepoch:{epoch}")

                    # epoch_lp = 0
                    for i, (x, _) in enumerate(dataloader):
                        # print(f"i:{i}")
                        model_pd.train()
                        pt = model_pd(Hx_dir={"Hx": x, 'edge_index': cinfo['tr']['edge_index']},
                                      bounds=bound,
                                      rate=rate,
                                      numofbyte=numofbyte,
                                      bandwidth=bandwidth).to(device)

                        optimizer.zero_grad()
                        model_pd.lagr.backward()

                        torch.nn.utils.clip_grad_norm_(model_pd.parameters(), 3)
                        optimizer.step()
                        model_pd.update(update_Step, epoch, i,)
                        lr_sch.step()

                        # 存取训练中变化

                        # epoch_lp += model_pd.l_p.item()
                        if not i%20:
                            print("**" * 20)
                            print(f"forward func Hx:{x[0:4, :]}\n pt:{pt[0:4, :]}")
                            l_p_list.append(model_pd.l_p.mean().item())
                            lagr_list.append(model_pd.lagr.mean().item())
                            print(f'\ntraining epoch:{epoch}, step:{i}',
                                  "\nmodel_pd.l_p.mean():", model_pd.l_p.mean().item(),
                                  "\nmodel_pd.l_d.mean():", model_pd.l_d.mean().item(),
                                  "\nmodel_pd.lagr.mean():", model_pd.lagr.mean().item(),
                                  "\nmodel_pd.lambdas:", model_pd.lambdas.items(),
                                  "\nmodel_pd.vars:",model_pd.vars.items(),
                                  # "\nmodel_pd.ef:", model_pd.ef.items(),
                                  # "\nmodel_pd.throughput.item()",model_pd.throughput.data,
                                  # "\nmodel_pd.temp_dict:", model_pd.temp_dict,
                                  )
                        print(f"epoch：{epoch}\t i:{i} \t global-step:{epoch*20+i}\t l-p:{model_pd.l_p.item()}")
                        writer.add_scalar(tag=f"{model_name}-l_p", scalar_value=model_pd.l_p.item(), global_step=epoch*20+i)
                        writer.add_scalar(tag=f"{model_name}-lagr", scalar_value=model_pd.lagr.item(),global_step=epoch * 20 + i)
            except KeyboardInterrupt:
                i = input('Save?').lower()
                if i == 't':
                    torch.save(model_pd, modelsavepath)
                    break
                elif i == 'f':
                    break
            torch.save(model_pd, modelsavepath)

            # # 验证部分
            # model_pd.eval()
            # with torch.no_grad():
            #     pt_val = model_pd(**valdata).to(device)
            #     # 计算验证集上的时延和中断概率
            #     delay_val = model_pd.l_p.mean().item()
            #     pout_val = Probabilty_outage(model_pd.get_utility_func(model_name), pt_val, factors_val, rate, NumK, len(pt_val), flag=True)

            #     valLogs[epoch] = {'delay_val': delay_val, 'pout_val': pout_val.mean().item()}
            #     print(f'Validation - Epoch {epoch}, Delay: {delay_val}, Pout: {pout_val.mean().item()}')
            #     writer.add_scalar(tag=f"{model_name}-val-delay", scalar_value=delay_val, global_step=epoch)
            #     writer.add_scalar(tag=f"{model_name}-val-pout", scalar_value=pout_val.mean().item(), global_step=epoch)




            #保存val验证集的log文件
            # import pickle
            # with open(valpath+'power_vallog.pk', 'wb') as f:
            #     pickle.dump(valLogs, f)

            from findFuncAnswer import equalAllocation
            equalAllocation(1000, factor, NumK, rate, bound, func=get_utility_func(model_name))



            #matplot draw graph
            x = np.linspace(start=0,stop=len(l_p_list),num=len(l_p_list))
            plt.figure(figsize=(16, 8))
            plt.subplot(1,2,1)
            plt.title(f"{model_name} epoch---delay", color='b')
            plt.xlabel("epoch")
            plt.ylabel("delay")
            plt.plot(x, l_p_list)
            if any(delay > 1. for delay in l_p_list):
                plt.ylim(0, 1.)
            
            plt.subplot(1,2,2)
            plt.title("epoch---Lagr", color='b')
            plt.xlabel("epoch")
            plt.ylabel("Lagr")
            plt.plot(x, lagr_list)
            
            plt.savefig(photo_save_path+f'\\{model_name}-NumK={NumK}-PDB={PDB}-equal_flag={equal_flag}.jpg')
            plt.show()

    #关闭tensorboard写入
    writer.close()


if __name__ =="__main__":
    factor=0.50
    train()
