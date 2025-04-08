import pickle
import numpy as np
# from train.findFuncAnswer import get_Equal_solution
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sea
import pandas as pd

#   this readdata.py file contain equal allocate policy
#   while drawGraph.py not
def draw_loss_graph():
    """
    绘制吞吐量曲线
    :return: 
    """

    def smooth(data, x='epoch', y='delay', weight=0.6):
        scalar = data[y].values
        last = scalar[100]
        smoothed = []
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return pd.DataFrame({x: data[x].values, y: smoothed, "name": data['name']})

    # event1 = 'train\tensorboardLOG\events.out.tfevents.1738766051.MIRAGE.9716.0'
    # event2 = 'train\tensorboardLOG\events.out.tfevents.1738809060.MIRAGE.15652.0'
    # event3 = 'train\tensorboardLOG\events.out.tfevents.1738809513.MIRAGE.22540.0'
    event1 = os.path.join(os.getcwd(), 'train', 'tensorboardLOG', 'events.out.tfevents.1738766051.MIRAGE.9716.0')
    event2 = os.path.join(os.getcwd(), 'train', 'tensorboardLOG', 'events.out.tfevents.1738809060.MIRAGE.15652.0')
    event3 = os.path.join(os.getcwd(), 'train', 'tensorboardLOG', 'events.out.tfevents.1738809513.MIRAGE.22540.0')

    if not os.path.exists(event1):
        print(f"文件不存在: {event1}")

    ea = event_accumulator.EventAccumulator(event1)
    ea.Reload()
    df1_train_loss = ea.scalars.Items('HARQ')
    df1_train_loss = [(i.step, i.value, 'HARQ') for i in df1_train_loss]
    df1_train_loss = pd.DataFrame(df1_train_loss, columns=['epoch', 'delay', 'name'])

    ea = event_accumulator.EventAccumulator(event2)
    ea.Reload()
    df2_train_loss = ea.scalars.Items('HARQ-IR')
    df2_train_loss = [(i.step, i.value, 'HARQ-IR') for i in df2_train_loss]
    df2_train_loss = pd.DataFrame(df2_train_loss, columns=['epoch', 'delay', 'name'])
    #
    ea = event_accumulator.EventAccumulator(event3)
    ea.Reload()
    df3_train_loss = ea.scalars.Items('HARQ-CC')
    df3_train_loss = [(i.step, i.value, 'HARQ-CC') for i in df3_train_loss]
    df3_train_loss = pd.DataFrame(df3_train_loss, columns=['epoch', 'delay', 'name'])
    

    df1 = smooth(df2_train_loss, weight=0.6)
    df2 = df1.loc[:, 'delay'].to_numpy()
    df2 = df2.reshape(1, -1).T
    np.savetxt('.\\tensorboardLOG\delay-itera-HARQ-IR4.csv', df2)
    #
    df1 = smooth(df3_train_loss , weight= 0.6)
    df2 = df1.loc[:, 'delay'].to_numpy()
    df2 = df2.reshape(1, -1).T
    np.savetxt('.\\tensorboardLOG\delay-itera-HARQ-CC4.csv', df2)


    df1 = smooth(df1_train_loss,weight=0.6)
    df2 = df1.loc[:, 'delay'].to_numpy()
    df2 = df2.reshape(1, -1).T
    np.savetxt('.\\tensorboardLOG\delay-itera-HARQ4.csv', df2)


    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    p = sea.lineplot(data=df1, x='epoch', y='delay', hue='name', style='name')
    p.set_xlabel("Number of iterations", fontsize=20)
    p.set_ylabel("$\eta $ (bps/Hz)", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

"""labels=["10DB","15DB","20DB","25DB"]"""
if __name__ == '__main__':

    draw_loss_graph()
    #draw_graph()






