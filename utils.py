import torch
import matplotlib.pyplot as plt

class Prioritized_Replay_Buffer():
    def __init__(self, state_shape, action_shape, size=2**15, batch_size=256, prioritization=0.9, prioritization_drop=0.9, device='cpu'):
        
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.prioritization = prioritization
        self.prioritization_drop = prioritization_drop

        self.states = torch.zeros((size,) + state_shape, device=device)
        self.actions = torch.zeros((size,) + action_shape, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.next_states = torch.zeros((size,) + state_shape, device=device)
        self.terminals = torch.zeros(size, dtype=torch.bool, device=device)
        self.weights = torch.zeros(size, device=device)+1e-10

        self.size = size
        self.elems = 0
        self.batch_size = batch_size

        self.device = device

        self.last_idx = None


    @torch.no_grad()
    def add_experience(self, states, actions, rewards, next_states, terminals, weights):

        states = states.reshape(-1,*self.state_shape)
        actions = actions.reshape(-1,*self.action_shape)
        rewards = rewards.reshape(-1)
        next_states = next_states.reshape(-1,*self.state_shape)
        terminals = terminals.reshape(-1)

        n = len(states)

        if self.elems + n <= self.size:
            self.states[self.elems : self.elems+n] = states
            self.actions[self.elems : self.elems+n] = actions
            self.rewards[self.elems : self.elems+n] = rewards
            self.next_states[self.elems : self.elems+n] = next_states
            self.terminals[self.elems : self.elems+n] = terminals
            self.weights[self.elems : self.elems+n] = weights
        else:
            
            i = self.size-self.elems

            if i>0:
                self.states[self.elems:] = states[:i].float()
                self.actions[self.elems:] = actions[:i].float()
                self.rewards[self.elems:] = rewards[:i].float()
                self.next_states[self.elems:] = next_states[:i].float()
                self.terminals[self.elems:] = terminals[:i].float()
                self.weights[self.elems :] = weights[:i].float()

            w = 1 / self.weights
            w = (w) * self.prioritization_drop + torch.full_like(self.weights[:self.elems], (1-self.prioritization_drop)/self.elems*w.sum())
            idx_replace = list(torch.utils.data.WeightedRandomSampler(w, n-i, replacement=False))

            if torch.any(self.rewards[idx_replace]==1):
                #print(torch.sum(self.rewards[idx_replace]==1))
                pass

            self.states[idx_replace] = states[i:].float()
            self.actions[idx_replace] = actions[i:].float()
            self.rewards[idx_replace] = rewards[i:].float()
            self.next_states[idx_replace] = next_states[i:].float()
            self.terminals[idx_replace] = terminals[i:].bool()
            self.weights[idx_replace] = weights[i:].float()

        if self.elems < self.size:
            self.elems = min(self.elems+n, self.size)

    
    def __len__(self,):
        return (self.elems + self.batch_size - 1) // self.batch_size  

    @torch.no_grad()
    def get(self, idx):
 
        start = min(idx*self.batch_size, self.elems)
        end = min((idx+1)*self.batch_size, self.elems)

        self.last_idx = torch.arange(start=start, end=end, dtype=torch.long)

        states = self.states[self.last_idx] 
        actions = self.actions[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]

        return states, actions, rewards, next_states, terminals
    
    @torch.no_grad()
    def get_high_priority_batch(self, repeat=True):

        n = self.batch_size if repeat else min(self.batch_size,self.elems)
        w = torch.exp(self.weights[:self.elems]-self.weights[:self.elems].max())
        w = w/w.sum() * self.prioritization + torch.full_like(self.weights[:self.elems],(1-self.prioritization)/self.elems) 
        self.last_idx = torch.tensor(list(torch.utils.data.WeightedRandomSampler(w, n, replacement=repeat)), device=self.device)

        states = self.states[self.last_idx] 
        actions = self.actions[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]
        weights = w[self.last_idx]

        return states, actions, rewards, next_states, terminals, weights
    
    @torch.no_grad()
    def update_weights(self, weights):

        assert self.last_idx is not None
        l=0.9
        self.weights[self.last_idx] = self.weights[self.last_idx]*l + weights*(1-l)



def update_target_model(model, target_model, decay=1e-2):
    model_dict = model.state_dict()
    target_model_dict = target_model.state_dict()
    for weight_key, target_weight_key in zip(model_dict.keys(),target_model_dict.keys()):
        target_model_dict[target_weight_key] = (1-decay)*target_model_dict[target_weight_key] + decay*model_dict[weight_key]
    target_model.load_state_dict(target_model_dict)