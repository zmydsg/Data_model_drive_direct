import torch
import torch.nn as nn
from utils import *
from model import *
import pandas as pd
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
import time
import warnings
# from torch_geometric.utils import is_undirected
from tempargs import *
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')
from torch.utils.data import TensorDataset
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

model_name_list = ["HARQ", "HARQ-CC", "HARQ-IR"]

PDBS = args.PDBs  # [19,  23]


def train_auto_pdb(eval=False):
    torch.autograd.set_detect_anomaly(eval)
    if eval:
        if not os.path.exists(val_path): os.mkdir(val_path)

    bounds = torch.log10(torch.tensor(bound)).to(device)
    pdbs = 10 ** (torch.tensor(PDBS, dtype=torch.float32) / 10)
    # data-process
    X, Y, cinfo = getdata(data_path, 0, NumK, device=device, equal_flag=equal_flag, eval=eval)
    dataset = TensorDataset(X['tr'], Y['tr'])
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    for model_name in model_name_list:
        # 初始化固定随机种子
        setup_seed(seed)
        # 生成存储控制台输出的文件路径
        logsrecord_name = print_save_path + f'{model_name}-Numk={NumK}-' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                        time.localtime()) + '.log'
        # 模型保存路径
        modelsavepath = model_save_path + f'{model_name}-NumK={NumK}_avg.pt'
        valpath = val_path + f'{model_name}-NumK={NumK}_'
        # 记录正常的 print 信息
        sys.stdout = Logger(logsrecord_name)
        # 记录 traceback 异常信息
        sys.stderr = Logger(logsrecord_name)

        if eval:
            valLogs = {}
            val = DataLoader(TensorDataset(X['val'], Y['val']), batch_size=32, shuffle=True)
            val_ = enumerate(val)
            valdata = {
                'Hx_dir': {'Hx': X['val'],
                           'edge_index': cinfo['val']['edge_index']},
                'bounds': bounds, 'rate': rate, 'numofbyte': numofbyte, 'bandwidth': bandwidth}

        print(f"\nbounds:{bounds}\tdb:{PDBS}\tPt_max:{X['tr'][0, -1]}")

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
        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100 * 20, gamma=0.4)
        l_p_list = []
        lagr_list = []
        gcnmodel.train()
        model_pd.train()

        try:
            # training
            for epoch in range(epochs):
                print("=" * 100)
                pdb = torch.randint(0, pdbs.shape[0], (50,), dtype=torch.long)  # 在每个 epoch 开始时重新生成 pdb

                for i, (x, _) in enumerate(dataloader):
                    with torch.no_grad():
                        x[:, -1] = pdbs[torch.randint(0, pdbs.shape[0], (x.shape[0],), dtype=torch.long, out=pdb)]
                        x[:, :NumK] = x[:, -1, None] / NumK
                        if eval and not i: print(x[0])
                    pt = model_pd(Hx_dir={"Hx": x, 'edge_index': cinfo['tr']['edge_index']},
                                  bounds=bound,
                                  rate=rate,
                                  numofbyte=numofbyte,
                                  bandwidth=bandwidth).to(device)

                    optimizer.zero_grad()
                    model_pd.lagr.backward()  #################################
                    torch.nn.utils.clip_grad_norm_(model_pd.parameters(), 3)
                    optimizer.step()
                    model_pd.update(update_Step, epoch, i, )
                    lr_sch.step()

                    # 存取训练中变化
                    if not i % 20:
                        l_p_list.append(model_pd.l_p.mean().item())
                        lagr_list.append(model_pd.lagr.mean().item())
                        print(f'\ntraining epoch:{epoch}\n',
                              "l_p:", model_pd.l_p.mean().item(),
                              " l_d:", model_pd.l_d.mean().item(),
                              " lagr:", model_pd.lagr.mean().item(),
                              )
                        writer.add_scalar(tag=f"{model_name}-l_p", scalar_value=model_pd.l_p.item(),
                                          global_step=epoch * 20 + i)
                        writer.add_scalar(tag=f"{model_name}-lagr", scalar_value=model_pd.lagr.item(),
                                          global_step=epoch * 20 + i)

                        # 批量记录数据
                        batch_size = x.shape[0]
                        current_pdbs = pdbs[pdb[:batch_size]].cpu().numpy().flatten()
                        pout_values = model_pd.pout.detach().cpu().numpy().flatten()

                        # 确保所有数组的长度一致
                        min_length = min(len(current_pdbs), len(pout_values), batch_size)

                        pout_data = {
                            'PDB': current_pdbs[:min_length].tolist(),
                            'Epoch': [epoch] * min_length,
                            'Step': [i] * min_length,
                            'Pout': pout_values[:min_length].tolist(),
                            'Lagr': [model_pd.lagr.item()] * min_length,
                            'L_p': [model_pd.l_p.item()] * min_length,
                            'L_d': [model_pd.l_d.item()] * min_length,
                            'Model': [model_name] * min_length
                        }

                        # 将数据写入CSV文件
                        csv_path = os.path.join(
                            model_save_path,
                            f'{model_name}_NumK={NumK}_training_data.csv'
                        )
                        df = pd.DataFrame(pout_data)
                        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

                if eval:
                    # 验证部分
                    model_pd.eval()
                    gcnmodel.eval()
                    gcnmodel.model.eval()
                    try:
                        valdata['Hx_dir']['Hx'] = next(val_)[1][0]
                    except StopIteration:
                        val_ = enumerate(val)
                        valdata['Hx_dir']['Hx'] = next(val_)[1][0]
                    with torch.no_grad():
                        pt_val = model_pd.forward(**valdata).to(device)
                        # 计算验证集上的时延和中断概率
                        delay_val = model_pd.l_p.item()
                        pout_val = model_pd.pout.mean().item()
                        valLogs[epoch] = {'delay_val': delay_val, 'pout_val': pout_val}
                        print(f'Delay: {delay_val}, Pout: {pout_val}')
                        writer.add_scalar(tag=f"{model_name}-val-delay", scalar_value=delay_val, global_step=epoch)
                        writer.add_scalar(tag=f"{model_name}-val-pout", scalar_value=pout_val, global_step=epoch)
                    model_pd.train()
                    gcnmodel.train()
                    gcnmodel.model.train()

        except KeyboardInterrupt:
            i = input().lower()
            if i == 't':
                torch.save(model_pd, modelsavepath)
                break
            elif i == 'f':
                break
        torch.save(model_pd, modelsavepath)

        if eval:
            # 保存val验证集的log文件
            import pickle
            with open(valpath + 'power_vallog.pk', 'wb') as f:
                pickle.dump(valLogs, f)

        # matplot draw graph
        x = np.linspace(start=0, stop=len(l_p_list), num=len(l_p_list))
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title(f"{model_name} epoch---delay", color='b')
        plt.xlabel("epoch")
        plt.ylabel("delay")
        plt.plot(x, l_p_list)
        if any(0 > delay > 0.5 for delay in l_p_list):
            plt.ylim(0, 0.5)

        plt.subplot(1, 2, 2)
        plt.title("epoch---Lagr", color='b')
        plt.xlabel("epoch")
        plt.ylabel("Lagr")
        plt.plot(x, lagr_list)
        if any(lagr > 10 for lagr in lagr_list):
            plt.ylim(None, 10)

        plt.savefig(photo_save_path + f'\\{model_name}-NumK={NumK}.jpg')
        plt.show()

        # 保存中断概率数据到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            model_save_path,
            f'{model_name}_NumK={NumK}_pout_data_{timestamp}.csv'
        )

        df = pd.DataFrame(pout_data)
        df.to_csv(csv_path, index=False)
        print(f"中断概率数据已保存到: {csv_path}")

        # 关闭tensorboard写入
        writer.close()


if __name__ == "__main__":
    factor = 0.50
    factors_val = args.factors
    reward = 0.8
    bound = 10e-2
    train_auto_pdb(False)
