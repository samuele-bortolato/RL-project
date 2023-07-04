import torch
import numpy as np
import time
import math



class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class Agent0(torch.nn.Module):
    def __init__(self,res=100):
        super().__init__()
        self.res = res
        self.grid = torch.nn.Parameter(torch.zeros((res,res)))

    def forward(self, x):
        
        pos_x = (x[:,0]*(self.res-1)).floor().long()
        pos_y = (x[:,1]*(self.res-1)).floor().long()
        value = self.grid[pos_x,pos_y]
        action = torch.rand((len(x),4), device=x.device)*2-1
        
        #print(torch.stack([pos_x, pos_y,value],1))
        return action, value[...,None]

class Agent1(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        action = self.action(x)
        value = self.value(x)
        return action, value
    
class Agent2(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        xr = x
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x)) + xr
        action = self.action(x)
        value = self.value(x)
        return action, value
    
class Agent3(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()
        self.n = 3
        self.l1 = torch.nn.Linear(in_features*(self.n*2+1), hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)
        self.K = torch.pow(1.5, torch.arange(6))
        

    def forward(self, x):
        e=torch.empty(x.shape+(self.n*2,), device=x.device)
        for i in range(self.n):
            e[:,:,2*i+0]=torch.sin(x*self.K[i]+i)
            e[:,:,2*i+1]=torch.cos(x*self.K[i]+i)
        x = torch.concat([x,e.reshape(x.shape[0],-1)],-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        action = self.action(x)
        value = self.value(x)
        return action, value
 

class Agent4(torch.nn.Module):
    def __init__(self,in_features, hidden_dim, levels=16, table_size=2048, channels=2, growith_factor = 1.3):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features+levels*channels, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)

        self.levels = levels
        self.table_size = table_size
        self.channels = channels
        self.T = torch.nn.Parameter(torch.randn(levels, table_size, channels))
        self.K = torch.nn.Parameter(torch.pow(growith_factor, torch.arange(levels)[:,None]), requires_grad=False)

        self.PRIMES = [2654436881, 5915587277, 1500450271, 3267000013, 5754853343,
         4093082899, 9576890767, 3628273133, 2860486313, 5463458053,
         3367900313, 5654500741, 5654500763, 5654500771, 5654500783,
         5654500801, 5654500811, 5654500861, 5654500879, 5654500889,
         5654500897, 5654500927, 5654500961, 5654500981, 5654500993,
         9999999967, 7654487179, 7654489553, 7654495087, 7654486423,
         7654488209, 8654487029, 8654489771, 8654494517, 8654495341]

    def forward(self, x):
        
        self.K = self.K.to(x.device)
        p = (x.cumsum(-1)[:,None]*self.K)
        c = p.frac().sort(-1)[0]
        R = (p.frac()[...,None]>=c[...,None,:]).float()
        R = torch.concat([R,torch.zeros(*R.shape[:-1],1, device=x.device)],-1)
        c = torch.concat([c,torch.ones(*c.shape[:-1],1, device=x.device)],-1)
        A = R + p.floor()[...,None]
        A -= torch.concat([torch.zeros_like(A[...,-1:,:]),A[...,:-1,:]],-2)
        c -= torch.concat([torch.zeros_like(c[...,-1:]),c[...,:-1]],-1)

        cidx = A[:,:,0].long() * self.PRIMES[0]
        for i in range(1, A.shape[2]):
            cidx = cidx ^ (A[:,:,i].int() * self.PRIMES[i])

        feats=[]
        for i in range(self.levels):
            t=self.T[i]
            feats.append(t[cidx[:,i] % self.table_size])
        f = torch.stack(feats,1)
        f = (f*c[...,None]/self.K[...,None]).sum(2)
        
        x = torch.concat([x, f.reshape(x.shape[0],-1)],-1)

        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        action = self.action(x)
        value = self.value(x)
        return action, value




    


    
config={
	"optimizer": {
		"otype": "Adam",
		"learning_rate": 1e-2,
		"beta1": 0.9,
		"beta2": 0.99,
		"epsilon": 1e-15,
		"l2_reg": 1e-6
	},
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 15,
		"base_resolution": 4,
		"per_level_scale": 1.3
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 2
	}
}

import tinycudann as tcnn

class Agent5(torch.nn.Module):
    def __init__(self, in_coordinates, hidden_dim=64, levels=16, table_bit_size=15, channels=2, growith_factor = 1.25, base_res=1):
        super().__init__()

        self.in_coordinates=in_coordinates

        config__encoding = {
            "otype": "HashGrid",
            "n_levels": levels,
            "n_features_per_level": channels,
            "log2_hashmap_size": table_bit_size,
            "base_resolution": base_res,
            "per_level_scale": growith_factor
        }
        config_nn = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 2
	    }

        self.encodings = torch.nn.ParameterList()
        for _ in range(in_coordinates):
            self.encodings.append(tcnn.Encoding(2, config__encoding))

        self.nn = tcnn.Network(in_coordinates * levels * channels, 5, config_nn)

    def forward(self, x):

        x = x.reshape(x.shape[0], self.in_coordinates, 2)

        enc = []
        for i in range(self.in_coordinates):
            enc.append( self.encodings[i](x[:,i]) )
        x = torch.concat(enc,-1)
        x = self.nn(x)

        return x[:,:-1], x[:,-1:]
    


class Agent6(torch.nn.Module):
    def __init__(self,in_features, hidden_dim, heads=64, heads_dim=2, levels=5):
        super().__init__()

        self.heads = heads
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.heads_dim = heads_dim
        self.levels= levels
        self.groups = 1

        self.l1 = torch.nn.Linear(in_features, hidden_dim)

        self.L_residual = torch.nn.ParameterList()
        for i in range(levels):
            lr1 = torch.nn.Linear(hidden_dim, hidden_dim)
            pr1 = torch.nn.Linear(hidden_dim, heads*heads_dim)
            lr2 = torch.nn.Linear((hidden_dim + heads)//self.groups, hidden_dim//self.groups)
            self.L_residual.append(torch.nn.ParameterList([lr1,pr1,lr2]))

        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 255)

    def forward(self, x):

        x = self.l1(x)

        for i in range(self.levels):
            lr1,pr1,lr2 = self.L_residual[i]
            xr = 2

            p1 = pr1(x).reshape(-1,self.heads, self.heads_dim) 
            d1 = torch.norm(p1, dim=-1) 
            x = torch.relu(lr1(x))
            x = torch.concat([x, d1],-1)

            x = x.reshape(-1,(self.hidden_dim + self.heads)//self.groups)
            x = torch.relu(lr2(x)).reshape(-1,self.hidden_dim)
            x = x + xr 

        action = self.action(x)
        value = self.value(x)
        return action, value
    


class Agent7(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.l1 = torch.nn.Linear(in_features, hidden_dim)

        self.l2 = torch.nn.Linear(hidden_dim*2, hidden_dim)

        self.l3 = torch.nn.Linear(hidden_dim*2, hidden_dim)

        self.action = torch.nn.Linear(hidden_dim*2, 2*2)
        self.value = torch.nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.concat([x.relu(), x.square()],-1)

        x = self.l2(x)
        x = torch.concat([x.relu(), x.square()],-1)
        
        x = self.l3(x)
        x = torch.concat([x.relu(), x.square()],-1)

        action = self.action(x)
        value = self.value(x)
        return action, value
    

class Agent8(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.l1(x).relu().square()
        x = self.l2(x).relu().square()
        action = self.action(x)
        value = self.value(x)
        return action, value
    

class Agent9(torch.nn.Module):
    def __init__(self,in_features, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.exp(-self.l1(x).square())
        x = torch.exp(-self.l2(x).square())
        action = self.action(x)
        value = self.value(x)
        return action, value
    

class Agent10(torch.nn.Module):
    def __init__(self, in_points, hidden_dim):
        super().__init__()
        self.in_points = in_points
        self.n = 0
        self.l1 = torch.nn.Linear((in_points*2+in_points*(in_points-1))*(self.n*2+1), hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.action = torch.nn.Linear(hidden_dim, 2*2)
        self.value = torch.nn.Linear(hidden_dim, 255)
        self.K = torch.pow(1.5, torch.arange(6))
        

    def forward(self, x):

        S = x.shape

        xs = x.reshape(S[0], self.in_points, 2)
        c = torch.cdist(xs,xs)
        c = c.reshape(S[0], -1)[:,1:].reshape(S[0], self.in_points-1, self.in_points+1)[:,:,:-1].reshape(S[0], -1)

        x = torch.concat([x,c],-1)

        e=torch.empty(x.shape+(self.n*2,), device=x.device)
        for i in range(self.n):
            e[:,:,2*i+0]=torch.sin(x*self.K[i]+i)
            e[:,:,2*i+1]=torch.cos(x*self.K[i]+i)
        x = torch.concat([x,e.reshape(x.shape[0],-1)],-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        action = self.action(x)
        value = self.value(x)
        return action, value
    

class Agent11(torch.nn.Module):
    def __init__(self, in_points, hidden_dim):
        super().__init__()
        self.n = 2
        self.in_points = in_points
        self.t_L = torch.nn.TransformerEncoderLayer(d_model=2*(1+self.n*2), nhead=1+self.n*2, dim_feedforward=hidden_dim)
        self.T = torch.nn.TransformerEncoder(self.t_L, 2)
        self.action = torch.nn.Linear(in_points*2*(1+self.n*2), 4)
        self.value = torch.nn.Linear(in_points*2*(1+self.n*2), 1)
        self.K = torch.pow(1.5, torch.arange(6))
        

    def forward(self, x):
        e=torch.empty(x.shape+(self.n*2,), device=x.device)
        for i in range(self.n):
            e[:,:,2*i+0]=torch.sin(x*self.K[i]+i)
            e[:,:,2*i+1]=torch.cos(x*self.K[i]+i)
        x = torch.concat([x.reshape(x.shape[0], self.in_points, 2),e.reshape(x.shape[0], self.in_points, -1)], -1)
        x = self.T(x)
        x = x.reshape(x.shape[0], -1)
        action = self.action(x)
        value = self.value(x)
        return action, value
    


class Agent12(torch.nn.Module):
    def __init__(self, in_features, heads=64, heads_dim=2):
        super().__init__()

        self.heads = heads
        self.in_features = in_features
        self.heads_dim = heads_dim

        self.p1 = torch.nn.Linear(in_features, heads*heads_dim)
        self.p2 = torch.nn.Linear(heads*(heads_dim+1), heads*heads_dim)
        self.p3 = torch.nn.Linear(heads*(heads_dim+1), heads*heads_dim)

        self.action = torch.nn.Linear(heads*(heads_dim+1), 2*2)
        self.value = torch.nn.Linear(heads*(heads_dim+1), 255)

    def forward(self, x):
        p1 = self.p1(x)
        d1 = torch.norm(p1.reshape(-1,self.heads, self.heads_dim), dim=-1) 
        x = torch.relu(p1)
        x = torch.concat([x, d1],-1)

        p2 = self.p2(x)
        d2 = torch.norm(p2.reshape(-1,self.heads, self.heads_dim), dim=-1) 
        x = torch.relu(p2)
        x = torch.concat([x, d2],-1)

        p3 = self.p3(x)
        d3 = torch.norm(p3.reshape(-1,self.heads, self.heads_dim), dim=-1) 
        x = torch.relu(p3)
        x = torch.concat([x, d3],-1)

        action = self.action(x)
        value = self.value(x)
        return action, value
    


class Agent13(torch.nn.Module):
    def __init__(self, in_points, hidden_dim, levels=3):
        super().__init__()
        self.in_points = in_points
        
        self.l1 = torch.nn.Linear((in_points*2), hidden_dim)
        self.p1 = torch.nn.Parameter(torch.randn((hidden_dim, in_points*2)))
        self.l2 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.p2 = torch.nn.Parameter(torch.randn((hidden_dim, hidden_dim*2)))
        self.action = torch.nn.Linear(hidden_dim*2, 2*2)
        self.value = torch.nn.Linear(hidden_dim*2, 255)
        self.eps = 1e-10
        

    def forward(self, x):

        l = self.l1(x).relu()
        p = (self.p1.exp()/x.shape[-1] @ x.square()[...,None] + self.eps).sqrt()[...,0]
        x = torch.concat([l,p],-1)

        l = self.l2(x).relu()
        p = (self.p2.exp()/x.shape[-1] @ x.square()[...,None] + self.eps).sqrt()[...,0]
        x = torch.concat([l,p],-1)
        
        action = self.action(x)
        value = self.value(x)
        return action, value
    



class Agent14(torch.nn.Module):
    def __init__(self, in_points, hidden_dim):
        super().__init__()
        self.in_points = in_points
        in_features = in_points*2
        self.l1 = torch.nn.Linear(in_features + in_features**2 + in_points*(in_points-1), hidden_dim)
        #self.l1 = torch.nn.Linear(in_features, hidden_dim)
        
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.l4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l5 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.action = torch.nn.Linear(hidden_dim, 25)
        self.value = torch.nn.Linear(hidden_dim, 255)
        with torch.no_grad():
            #self.action.weight[:] *= 0.1
            #self.action.bias[:] = 1/hidden_dim
            self.value.weight[:] = 1
            self.value.bias[:]= 0
            pass

    def forward(self, x):
        
        S = x.shape

        x2 = x[:,None]*x[:,:,None]/2

        xs = x.reshape(S[0], self.in_points, 2)
        c = torch.cdist(xs,xs)
        c = c.reshape(S[0], -1)[:,1:].reshape(S[0], self.in_points-1, self.in_points+1)[:,:,:-1].reshape(S[0], -1)


        x = torch.concat([x,x2.flatten(1),c],1)#,x3.flatten(1)
        x = torch.relu(self.l1(x))

        xr = x
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x)) + xr

        xr = x
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x)) + xr

        action = self.action(x)
        value = self.value(x)
        return action, value

