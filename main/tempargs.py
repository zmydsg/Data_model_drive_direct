import argparse

import numpy as np
import torch
parser = argparse.ArgumentParser(description="paramaters of system")

parser.add_argument('--epochs', type=int, default=500,
                    help="training epoch")
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help="adam learning rate")
parser.add_argument('--learning_rate2', type=float, default=1e-3,
                    help="pout dual element step")
parser.add_argument('--learning_rate3', type=float, default=5e-5,
                    help="power dual element step")
parser.add_argument('--Bounds', type=dict,
                    default={10: 10e-3, 12: 10e-3, 14: 10e-3, 16: 10e-3,
                             18: 10e-3, 20: 10e-3, 22: 10e-3, 24: 10e-3,
                             15: 10e-3, 20: 10e-3, 25: 10e-3, 30: 10e-3},
                    
                    help="outage bounds")
parser.add_argument('--rate', type=int, default=2,
                    help="transmission rate")
parser.add_argument('--numofbyte', type=int, default=10e6,
                    help="total bits to tranmission")
parser.add_argument('--bandwidth', type=int, default=10e7,
                    help="band width to tranmission")
parser.add_argument('--equal_flag',type= bool , default= True,
                    help="allocate power policy")
parser.add_argument('--dropout', type=float,default=0,
                    help="dropout pro")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else 'cpu',
                    help="use cpu or gpu")

parser.add_argument('--PDB', type=int, default=15,
                   
                    help="power in DB")
parser.add_argument('--seed', type=int,default=42, #2022 ,
                    help="random seed")
parser.add_argument(('--train_seed'), type = int, default= 40,
                    help ="seed of generate training data ")
parser.add_argument(('--val_seed'), type = int, default= 42, #50
                    help ="seed of generate training data ")
parser.add_argument('--NumK', type=int, default=3,
                    help="transmission times")
parser.add_argument('--in_size', type = int, default=1,
                    help='input feature size')
parser.add_argument('--out_size', type = int, default=1,
                    help='output feature size')
parser.add_argument('--inter', default=[
        {'inter_size': [16, 32, 16, 2],
            'inter_activation': ['elu', 'elu', 'elu', 'elu',  'linear'],
         }],
                    help='hidden layer structer')
parser.add_argument('--model_name', type=str, default="HARQ-IR",
                    help='different model name')
args = parser.parse_args()
