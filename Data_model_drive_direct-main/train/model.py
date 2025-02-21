import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import train.utils as utils
import torch.optim as optim
import numpy as np
from train.generateData import generate1
from torch_geometric.utils import dense_to_sparse

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

    def forward(self, x, edge_index, edge_weights):         # (13)
        for i, (hidden, activa) in enumerate(zip(self.hidden, self.activs)):
            # print(x.shape, edge_index.shape, edge_weights.shape, hidden)
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

    def forward(self, Hx, edge_index=None, **extra):        # Vl/p_k[l,nonzero], W[l,nonzero], factor[l,1], p_max[l,1]

        # process the index
        if edge_index is None:
            edge_index = self.ei
        # if self.training:
        self._ei_batch = utils.edge_index_batch(edge_index, Hx.shape[0], self.size,  Hx.device)
        # else:
        #     self._ei_batch = edge_index

        p_init = Hx[:, :self.size].reshape(-1, 1)

        edge_weights_batch = Hx[:, self.size:-2].reshape(-1)        # nonzero
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

        base1 = torch.ones(1, device=device, requires_grad=False)  # default: requires_grad=False      # lambda pout
        base2 = torch.zeros(1, device=device, requires_grad=False)
        base3 = torch.ones(1, device=device, requires_grad=False) # default: requires_grad=False       # lambda power

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

        pt = self.model.forward(**Hx_dir)

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

        self.l_p = l_p_1.mean()    # tao                    (14)

        l_d = 0.
        # print(f"bounds:{bounds , type(bounds)}")
        for kc in self.constraints:
            if kc == "pout":        # P_out,K               (14)
                ef = torch.log10(pout[:, -1])-bounds    # log(re)
                self.ef1 = ef.detach()
                self.ef[kc] = ef.mean(dim=0, keepdim=True)
                
            else:                   # p_avg                 (14)
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



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.layer_3 = nn.Linear(32, action_dim)
        nn.init.normal_(self.layer_3.weight, 0., 0.1)
        nn.init.constant_(self.layer_3.bias, 0.1)

        self.max_action = torch.tensor(max_action)

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        a = self.max_action * torch.sigmoid(self.layer_3(a))

        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 64)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.layer_3 = nn.Linear(32, 1)
        nn.init.normal_(self.layer_3.weight, 0., 0.1)
        nn.init.constant_(self.layer_3.bias, 0.1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.layer_1(sa))
        q = torch.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, action_dim))

    def add(self, state, next_state, action, reward, done):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr += 1
        if self.ptr == self.max_size:
            self.ptr = 0
        elif self.size < self.max_size:
            self.size += 1
    def clean(self):
        self.ptr=self.size=0

    def get(self, idx):
        return (
            self.state[idx],
            self.next_state[idx],
            self.action[idx],
            self.reward[idx],
            self.done[idx]
        )

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.get(idx)

    def item(self, batch_size):
        for i in range(0, len(self), batch_size):
            idx = (np.arange(i, i+batch_size, dtype=np.int32) + self.ptr) % len(self)
            yield self.get(idx)

    def __len__(self):
        return self.size


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, discount=0.9, tau=0.005, lr=1e-6):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau

    def __call__(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def forward(self, state, action, next_state, done, reward):
        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward.sum() / (reward > 0).detach().sum() + (done * self.discount * target_Q).detach()
        # Get current Q estimate
        current_Q = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, action).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, replay_buffer, batch_size=32, iterations=32):
        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)
            self.forward(state, action, next_state, done, reward)
        self.update()

    def train_(self, buf):
        for x, y, u, r, d in buf:
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)
            self.forward(state, action, next_state, done, reward)
        self.update()

    def update(self):
        tau = self.tau
        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))



class Environment:
    def __init__(self, model_name, factor, NumK, reward, pmax, rate, numofbyte, bandwidth):
        self.outage_cal_func = utils.get_utility_func(model_name)
        self.factor = factor
        self.NumK = NumK
        self.rate = rate
        self.rate2 = rate*rate
        self.Nb = numofbyte
        self.B = bandwidth
        self.roi = numofbyte / rate / bandwidth
        self.hk = None
        self.H = None
        self.reward = reward
        self.pmax = pmax

    def __call__(self, action, k, next_state):
        # 判断本次传输是否成功
        self.outage_cal_func(next_state, action, self.factor, k, self.rate2, k)
        Pk = np.random.uniform()
        action = action[k-1].detach()
        SNR = action * self.hk[k-1]**2
        C = self.B * np.log2(1 + SNR)
        Flag = True    # 判断该数据包是否传输成功
        if Pk <= next_state[k-1] or self.rate > C or self.pmax < action: # 本次传输失败
            reward = 0
            if k == self.NumK: # 达到最大传输次数也就是该数据包传输失败
                next_state[:] = 0
                done = True    # 判断该数据包是否传输完成
                Flag = False
            else: # 没有达到最大传输次数
                next_state[k-1] = action
                done = False
        else: # 本次传输成功
            reward = self.reward / np.sqrt(k)
            next_state[k-1] = action
            next_state[k:] = 0
            done = True

        return reward, done, Flag


class HybridModel(nn.Module):
    def __init__(self, model_name, gcn_model: GCN, ddpg_model: DDPG, environment, Numk, PoutFactor, constraints,
                 train_size, device, learning_rates):
        super(HybridModel, self).__init__()
        self.device = device
        self.gcn = gcn_model
        self.optimizer = torch.optim.Adam(gcn_model.parameters(), lr=learning_rates['init'])
        # 动态学习率，调整learning rate大小
        self.lr_sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100*10, gamma=0.4)
        self.learning_rate = learning_rates
        self.agent = ddpg_model
        self.Numk = Numk
        self.environment = environment
        self.constraints = constraints
        self.buf = ReplayBuffer(Numk, 1, train_size)
        assert 'power' in constraints

        base1 = torch.ones(1, device=device, requires_grad=False)  # default: requires_grad=False      # lambda pout
        base2 = torch.zeros(1, device=device, requires_grad=False)
        base3 = torch.ones(1, device=device, requires_grad=False)  # default: requires_grad=False       # lambda power

        self.outage_cal_func_ = utils.get_utility_func_(model_name)
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
        self.LTAT = 0.
        self.PoutK = torch.zeros(Numk+1, dtype=torch.float32, device=device, requires_grad=False)
        self.delayK = self.PoutK.clone()
        self.PoutFactor = PoutFactor
        self.delay = 0.
        self.next_state = torch.zeros(Numk, dtype=torch.float32, device=device, requires_grad=False)
        self.state = self.next_state.clone()
        self.kwargs = {}

    def __call__(self, pin, H, ):
        state = self.gcn(np.concatenate(pin, H)[None])
        next_state = self.next_state.clone()
        for k in range(1, 1+self.Numk):
            action = self.agent(state)
            yield state, action, k, next_state
            self.agent.forward(state, action, next_state, **self.kwargs)
            if self.kwargs['done']:
                break

    def forward(self, Hx, edge_wight):
        self.state = state = self.gcn(Hx, edge_wight)
        NumK = state.shape[1]
        NumE = state.shape[0]
        next_state = state.clone()
        done = torch.zeros(NumE, dtype=torch.int16, device=state.device, requires_grad=False)
        reward = torch.zeros_like(state)
        Pout = torch.zeros_like(self.PoutK)
        for k in range(1, 1+NumK):
            action = self.agent.actor.forward(next_state)
            print(action.shape)
            for it in range(NumE):
                if done[it]:
                    continue
                self.environment.factor = Hx[it, -2].detach()
                self.environment.hk = 1. + 2 * (self.environment.factor ** np.arange(self.Numk)) * np.sqrt(1. - 2 * (self.environment.factor ** (2*np.arange(self.Numk))))
                reward[it, k-1], done_, Flag = self.environment(action[it], k, next_state[it])
                done[it] = done_
                self.buf.add(state[it].detach().numpy(), next_state[it].detach().numpy(), action[it].detach().numpy(), reward[it].detach().numpy(), done_)
                if done_:
                    if Flag:
                        Pout[:k] += 1
                    else:
                        Pout[:k-1] += 1
        self.PoutK *= 1. - self.PoutFactor
        self.PoutK += Pout * (self.PoutFactor / NumE)
        return Pout

    def train_(self, Hx, edge_wight, bounds, rate, numofbyte, bandwidth, branch=1):
        self.train()
        self.gcn.train()
        Pout = self.forward(Hx, edge_wight)

        pt_max = torch.reshape(Hx[:, -1], (-1, 1))

        # outage probability
        self.delay = (1 - self.PoutFactor) * self.delay + self.PoutFactor * Pout.mean() * numofbyte / (bandwidth * rate)
        pout = self.PoutK.detach()
        throughput = utils.through_output(pout, rate, self.Numk)
        self.LTAT = throughput.mean()
        l_p_1 = utils.delay_compute(throughput, numofbyte, bandwidth)
        self.l_p = l_p_1.mean()  # tao                    (14)
        print(self.delay.detach(), self.l_p.detach())

        for it in range(branch):
            l_d = 0.
            for kc in self.constraints:
                if kc == "pout":  # P_out,K               (14)
                    ef = torch.log10(pout[:, -1]) - bounds  # log(re)
                    self.ef[kc] = ef.mean(dim=0, keepdim=True)
                else:  # p_avg                 (14)
                    l_d_1 = utils.compute_p(pout, self.state)
                    l_d_2 = l_d_1 - pt_max
                    self.ef[kc] = l_d_2.mean(dim=0)

                self.vars[kc] = self.ef[kc].detach()
                l_d += self.lambdas[kc] @ self.ef[kc].T

            self.l_d = l_d
            self.lagr = self.delay + self.l_d     #####

            self.agent.train_(self.buf.sample(Hx.shape[0]))

            self.optimizer.zero_grad()
            self.lagr.backward()  #################################
            self.update()
            # self.agent.actor_optimizer.l
        return self.state

    def update(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
        self.optimizer.step()
        for kc in self.constraints:
            self.lambdas[kc] = torch.relu(self.lambdas[kc] + self.learning_rate[kc] * self.vars[kc])
        self.detach()
        self.lr_sch.step()

    def detach(self):
        for kc in self.constraints:
            self.vars[kc].detach_()
            self.lambdas[kc].detach_()


class MultiModel(nn.Module):
    def __init__(self, gcn_model, pd_models, NumN, NumK, device):
        super(MultiModel, self).__init__()
        self.device = device
        self.gcn_model = gcn_model
        gcn_model.train()
        self.pd_models = pd_models
        for pd in pd_models: pd.eval()
        self.NumN = NumN
        self.NumK = NumK
        self.Hij = torch.ones((NumN, NumN), dtype=torch.float32, requires_grad=False)
        self.Hx = None
        self.pas=None
        self._zero = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self._one = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.delay = self._zero.clone()
        self.LTAT = self._one.clone()
        self.SINR = self._one.clone()
        self.l_p = self._zero.clone()
        self.active = nn.Tanh()

    def forward(self, Hx_dirs, edge_index_, bounds, delays, pms, avg, rate, numofbyte, bandwidth):
        Hx_dirs = Hx_dirs.reshape(-1, self.NumN, self.NumK+self.NumK*(self.NumK+1)//2+2)
        factors = Hx_dirs[:, :, -2]
        NumE = Hx_dirs.shape[0]
        if self.Hx is None or self.Hx.shape[0] != NumE:
            self.Hx = torch.zeros((NumE, self.NumN*(self.NumN+1)+2), dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            for e in range(NumE):
                generate1(factors[e].detach().numpy(), self.NumK, self.Hij)
                edge_index, self.Hx[e, self.NumN: -2] = dense_to_sparse(self.Hij)
            edge_index = edge_index.to(self.device)
            for n in range(self.NumN):
                self.Hx[:, n] = pms[n]
        pt = self.gcn_model(self.Hx, edge_index)
        # torch.clip(pt, 0, 1, out=pt)

        with torch.no_grad():
            Hx_dirs[:, :, :self.NumK] = pt[:, :, None] / self.NumK
            for n in range(self.NumN):
                Hx_dirs[:, n, -1] = pt[:, n] * pms[n]

        delay = self._zero.clone()
        ltat = self._one.clone()
        sinr = self._one.clone()
        pts = []
        if self.pas is None or self.pas.shape[0] != NumE:
            self.pas = torch.zeros((NumE, self.NumN), dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            for n in range(self.NumN):
                pt = self.pd_models[n].model(Hx=Hx_dirs[:, n], edge_index=edge_index_).to(self.device)
                pts.append(torch.clip(pt, 0, pms[n]*avg, out=pt))
                self.pas[:, n] = pt.mean(dim=1)
        for n in range(self.NumN):
            pout = utils.Probabilty_outage(self.pd_models[n].outage_cal_func, pts[n], factors[:, n], rate, self.NumK, NumE)
            pout = pout * utils.SINR_D(self.pas, n, self.NumK)
            LTAT = utils.through_output(pout, rate, self.NumK)
            sinr = sinr * utils.compute_p(pout, pts[n])
            delay = delay + utils.delay_compute(LTAT, numofbyte, bandwidth).mean()
            ltat = ltat * LTAT
        self.delay = delay / self.NumN
        self.SINR = sinr.mean()
        self.LTAT = ltat.mean()
        self.l_p = self.LTAT
        return pts

class MultiModel0(nn.Module):
    def __init__(self, gcn_model, pd_models, NumN, NumK, device):
        super(MultiModel0, self).__init__()
        self.device = device
        self.gcn_model = gcn_model
        gcn_model.train()
        self.pd_models = pd_models
        for pd in pd_models: pd.eval()
        self.NumN = NumN
        self.NumK = NumK
        self.Hij = None
        self.Hx = None
        self._zero = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.delay = self._zero.clone()
        self.square = self._zero.clone()
        self.l_p = self._zero.clone()
        self.active = nn.Tanh()

    def forward(self, Hx_dirs, edge_index_, bounds, delays, rate, numofbyte, bandwidth):
        # Hx_dirs = Hx_dirs.reshape(-1, self.NumN, self.NumK+self.NumK*(self.NumK+1)//2+2)  avoid change input
        factors = Hx_dirs[:, :, -2]
        NumE = factors.shape[0]
        if self.Hij is None or self.Hx.shape[0] != NumE:
            self.Hij = torch.empty((NumE, self.NumN, self.NumN), dtype=torch.float32, requires_grad=False)
            self.Hx = torch.empty((NumE, self.NumN*(self.NumN+1)+2), dtype=torch.float32, requires_grad=False)
        for n in range(NumE):
            Hij = generate1(factors[n].detach().numpy(), self.NumK, self.Hij[n])
            edge_index, self.Hx[n, self.NumN: -2] = dense_to_sparse(Hij)
            # print(edge_index)
            self.Hx[n, :self.NumN] = Hx_dirs[n, :, -1]
        edge_index = edge_index.to(self.device)
        pt = self.gcn_model(self.Hx, edge_index)
        Hx_dirs[:, :, :self.NumK] = pt[:, :, None] / self.NumK

        delay = self._zero.clone()
        square = self._zero.clone()
        pts = []
        for n in range(self.NumN):
            with torch.no_grad():
                pt = self.pd_models[n](Hx_dir={'Hx': Hx_dirs[:, n], 'edge_index': edge_index_},
                                bounds=bounds[n],
                              rate=rate,
                              numofbyte=numofbyte,
                                bandwidth=bandwidth).to(self.device)
                pts.append(pt)
            delay += -self.pd_models[n].delay
            square += (self.pd_models[n].l_p + delays[n])**2
        self.delay[0] = delay / self.NumN
        self.square[0] = square / (self.NumN-1)
        l_p = self.delay - self.square
        self.l_p[0] = l_p#self.l_p - self.active(self.l_p - l_p)
        return pts
