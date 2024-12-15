import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import utils


class BasicGcn(torch.nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super(BasicGcn, self).__init__()
        # construct middle  for activation layer
        activations = {'elu': nn.ELU(), 'relu': nn.ReLU(), 'selu': nn.SELU(),
                       'sigmoid': nn.Sigmoid(), 'LeakyReLU': nn.LeakyReLU(),'tanh':nn.Tanh(),'none': nn.Identity(), 'linear': nn.Linear(out_size, out_size)}
        activs = ['relu'] * len(h_sizes) if activs is None else activs

        self.activs = nn.ModuleList()
        for a in activs:
            self.activs.append(activations[a])

        self.dp = dropout

        # modulelist container for hidden layers
        hidden_sizes = [in_size, *h_sizes, out_size] #[1, 4, 8, 16, 4, 1, 1]
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(GCNConv(hidden_sizes[k], hidden_sizes[k + 1]))

    def forward(self, x, edge_index, edge_weights):
        for i, (hidden, activa) in enumerate(zip(self.hidden, self.activs)):
            x = hidden(x, edge_index, edge_weight=edge_weights)
            x = F.dropout(x, p=self.dp, training=self.training)
            x = activa(x)
        return x

class GCN(nn.Module):
    """
        GCN with channel parameters as edge weights and pt as node signals
    """
    def __init__(self, in_size, out_size, inter_size, inter_activation=None, dropout=0.5, **extra):
        super().__init__()
        self.model = BasicGcn(in_size, out_size, inter_size, inter_activation, dropout)
        self.size = extra['NumK']
        self.ei = extra['edge_index']
        self._ei_batch = None

    def forward(self, Hx, edge_index=None, **extra):

        # process the index
        if edge_index is None:
            edge_index = self.ei
        if self.training:
            self._ei_batch = utils.edge_index_batch(edge_index, Hx.shape[0], self.size,  Hx.device)
        else:
            self._ei_batch = edge_index


        p_init = Hx[:, :self.size].reshape(-1, 1)
        p_min = p_init[0]

        edge_weights_batch = Hx[:, self.size:-2].reshape(-1)
        pt = self.model(p_init, self._ei_batch, edge_weights_batch).reshape(-1, self.size)
        
        return pt


class PrimalDualModel(nn.Module):
    def __init__(self, model_name, model, Numk, constraints, device):
        super(PrimalDualModel, self).__init__()
        self.device = device
        self.model = model
        self.Numk = Numk
        self.constraints = constraints
        assert 'power' in constraints

        base1 = torch.ones((1), device=device, requires_grad=False)  # default: requires_grad=False      # lambda pout
        base2 = torch.zeros((1), device=device, requires_grad=False)
        base3 = torch.ones((1), device=device, requires_grad=False) # default: requires_grad=False       # lambda power

        self.outage_cal_func = utils.get_utility_func(model_name)
        self.vars = {}
        self.lambdas = {}
        self.temp_dict = {}
        self.ef = {}

        for kc in constraints:
            self.vars[kc] = base2.clone()
            if kc == 'pout':
                self.lambdas[kc] = base1.clone()
            else:
                self.lambdas[kc] = base3.clone()

        self.l_p = 0.
        self.l_d = 0.
        self.lagr = 0.
        self.mid0 = []

        print(f"model init: \nlambdas:{self.lambdas},\nvars:{self.vars}\n")

    def forward(self, Hx_dir, bounds, rate, numofbyte, bandwidth):
        Hx = Hx_dir['Hx']

        pt = self.model(**Hx_dir)

        pt_max = torch.reshape(Hx[:,-1],(-1,1))
        factors = Hx[:,-2]

        NumK = pt.shape[1]
        NumE = pt.shape[0]

        # outage probability
        pout = utils.Probabilty_outage(self.outage_cal_func, pt, factors, rate, NumK, NumE)
        # print(pout[:5, :])
        self.pout = pout.detach()
        # delay //throught_output
        throughput = utils.through_output(pout, rate, NumK)
        self.throughput = throughput.detach()
        l_p_1 = utils.delay_compute(throughput, numofbyte, bandwidth)
        self.delay = l_p_1.detach()

        self.l_p = l_p_1.mean()

        l_d = 0.
        # print(f"bounds:{bounds , type(bounds)}")
        for kc in self.constraints:
            if kc == "pout":
                ef = torch.log10(pout[:, -1])-bounds
                self.ef1 = ef.detach()
                self.ef[kc] = ef.mean(dim=0, keepdim=True)
                
            else:
                l_d_1 = utils.compute_p(pout, pt)
                l_d_2 = l_d_1-pt_max
                self.ef2 = l_d_2.detach()
                self.ef[kc] = l_d_2.mean(dim=0)
                
            self.vars[kc] = self.ef[kc].detach()
            
            l_d += self.lambdas[kc] @ self.ef[kc].T

        # self.l_d = torch.squeeze(l_d)
        self.l_d = l_d

        self.lagr = self.l_p + self.l_d
        
        return pt

#使用给定步长 stepsizes 来更新对偶变量 lambdas[kc]，保证非负性 (使用torch.relu)。
    def update(self,  stepsizes, epoch= None, i= None):

        for kc in self.constraints:
            # if kc == "power" and epoch%100==99 and i %20 ==0 :
            #     stepsizes[kc] *=0.5
            #     print(f"stepsizes:{stepsizes}")
            #     print(f"update function:\t kc:{kc},self.vars[kc]:{self.vars[kc]},stepsizes:{stepsizes} ",)
            self.lambdas[kc] = torch.relu(self.lambdas[kc] + stepsizes[kc] * self.vars[kc])

        self.detach()

    def detach(self):
        for kc in self.constraints:
            self.vars[kc].detach_()
            self.lambdas[kc].detach_()


        