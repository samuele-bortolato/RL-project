import torch
import numpy as np
import time
import math


class Agent(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, levels=2, input_type='complete', actor=True, critic=True, return_hidden=True, action_dimension=25, value_dimension=255, ):
        super().__init__()
        self.in_points = in_dim//2

        self.actor = actor
        self.critic = critic
        self.return_hidden = return_hidden

        self.input_type = input_type
        if input_type=='base':
            self.l1 = torch.nn.Linear(in_dim, hidden_dim)
        elif input_type=='augmented':
            self.l1 = torch.nn.Linear(in_dim + in_dim**2, hidden_dim)
        elif input_type=='complete':
            self.l1 = torch.nn.Linear(in_dim + in_dim**2 + self.in_points*(self.in_points-1), hidden_dim)
        else:
            assert False, "Unknown input configuration"
        
        self.levels = levels
        self.L_residual = torch.nn.ParameterList()
        for _ in range(levels):
            lr1 = torch.nn.Linear(hidden_dim, hidden_dim)
            lr2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.L_residual.append(torch.nn.ParameterList([lr1,lr2]))

        if actor:
            self.action = torch.nn.Linear(hidden_dim, action_dimension)

        if critic:
            self.value = torch.nn.Linear(hidden_dim, value_dimension)

    def forward(self, x):

        if self.input_type == 'augmented':

            S = x.shape

            x2 = x[:,None]*x[:,:,None]/2

            x = torch.concat([x,x2.flatten(1)],1)

        elif self.input_type == 'complete':

            S = x.shape

            x2 = x[:,None]*x[:,:,None]/2

            xs = x.reshape(S[0], self.in_points, 2)
            c = torch.cdist(xs,xs)
            c = c.reshape(S[0], -1)[:,1:].reshape(S[0], self.in_points-1, self.in_points+1)[:,:,:-1].reshape(S[0], -1)

            x = torch.concat([x,x2.flatten(1),c],1)
        
        x = torch.relu(self.l1(x))

        for i in range(self.levels):
            lr1,lr2 = self.L_residual[i]
            xr = x
            x = torch.relu(lr1(x))
            x = torch.relu(lr2(x)) + xr

        if self.actor and self.critic:
            return self.action(x), self.value(x), x if self.return_hidden else None
        
        elif self.critic:
            return None, self.value(x), x if self.return_hidden else None
        
        elif self.actor:
            return self.action(x), None, x if self.return_hidden else None