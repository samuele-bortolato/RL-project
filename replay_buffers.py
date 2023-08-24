import torch
import matplotlib.pyplot as plt


class Replay_Buffer():
    def __init__(   self, 
                    state_shape, 
                    action_shape, 
                    size=2**15, 
                    batch_size=256, 
                    device='cpu'):
        
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.states = torch.zeros((size,) + state_shape, device=device)
        self.actions = torch.zeros((size,) + action_shape, device=device)
        self.action_probs = torch.zeros(size, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.next_states = torch.zeros((size,) + state_shape, device=device)
        self.terminals = torch.zeros(size, dtype=torch.bool, device=device)

        self.size = size
        self.elems = 0
        self.idx = 0
        self.batch_size = batch_size

        self.device = device

        self.last_idx = None


    @torch.no_grad()
    def add_experience(self, states, actions, action_probs, rewards, next_states, terminals):

        states = states.reshape(-1,*self.state_shape)
        actions = actions.reshape(-1,*self.action_shape)
        action_probs = action_probs.reshape(-1)
        rewards = rewards.reshape(-1)
        next_states = next_states.reshape(-1,*self.state_shape)
        terminals = terminals.reshape(-1)

        n = len(states)

        if self.idx + n <= self.size:
            
            self.states[self.idx : self.idx+n] = states
            self.actions[self.idx : self.idx+n] = actions
            self.action_probs[self.idx : self.idx+n] = action_probs
            self.rewards[self.idx : self.idx+n] = rewards
            self.next_states[self.idx : self.idx+n] = next_states
            self.terminals[self.idx : self.idx+n] = terminals

        else:

            i = self.size-self.idx

            self.states[self.idx:] = states[:i]
            self.actions[self.idx:] = actions[:i]
            self.action_probs[self.idx:] = action_probs[:i]
            self.rewards[self.idx:] = rewards[:i]
            self.next_states[self.idx:] = next_states[:i]
            self.terminals[self.idx:] = terminals[:i]

            self.states[:n-i] = states[i:]
            self.actions[:n-i] = actions[i:]
            self.action_probs[:n-i] = action_probs[i:]
            self.rewards[:n-i] = rewards[i:]
            self.next_states[:n-i] = next_states[i:]
            self.terminals[:n-i] = terminals[i:]

        self.idx = (self.idx + n) % self.size

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
        action_probs = self.action_probs[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]

        return states, actions, action_probs, rewards, next_states, terminals
    
    @torch.no_grad()
    def get_batch(self, repeat=True):

        n = self.batch_size if repeat else min(self.batch_size,self.elems)

        if repeat:
            self.last_idx = torch.randint(0,self.elems,(n,))
        else:
            self.last_idx = torch.randperm(self.elems)[:n]

        states = self.states[self.last_idx] 
        actions = self.actions[self.last_idx]
        action_probs = self.action_probs[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]

        return states, actions, action_probs, rewards, next_states, terminals, None


class Prioritized_Replay_Buffer():
    def __init__(   self, 
                    state_shape, 
                    action_shape, 
                    size=2**15, 
                    batch_size=256, 
                    prioritization=1,
                    prioritized_drop = False, 
                    prioritization_drop=1, 
                    device='cpu'):
        
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.prioritization = prioritization
        self.prioritized_drop = prioritized_drop
        self.prioritization_drop = prioritization_drop

        self.states = torch.zeros((size,) + state_shape, device=device)
        self.actions = torch.zeros((size,) + action_shape, device=device)
        self.action_probs = torch.zeros(size, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.next_states = torch.zeros((size,) + state_shape, device=device)
        self.terminals = torch.zeros(size, dtype=torch.bool, device=device)
        self.weights = torch.zeros(size, device=device)+1e-10

        self.size = size
        self.elems = 0
        self.idx = 0
        self.batch_size = batch_size

        self.device = device

        self.last_idx = None


    @torch.no_grad()
    def add_experience(self, states, actions, action_probs, rewards, next_states, terminals, weights):

        states = states.reshape(-1,*self.state_shape)
        actions = actions.reshape(-1,*self.action_shape)
        action_probs = action_probs.reshape(-1)
        rewards = rewards.reshape(-1)
        next_states = next_states.reshape(-1,*self.state_shape)
        terminals = terminals.reshape(-1)

        n = len(states)

        if self.prioritized_drop:
            if self.elems + n <= self.size:
                self.states[self.elems : self.elems+n] = states
                self.actions[self.elems : self.elems+n] = actions
                self.action_probs[self.elems : self.elems+n] = action_probs
                self.rewards[self.elems : self.elems+n] = rewards
                self.next_states[self.elems : self.elems+n] = next_states
                self.terminals[self.elems : self.elems+n] = terminals
                self.weights[self.elems : self.elems+n] = weights
            else:

                i = self.size-self.elems

                if i>0:
                    self.states[self.elems:] = states[:i]
                    self.actions[self.elems:] = actions[:i]
                    self.action_probs[self.elems:] = action_probs[:i]
                    self.rewards[self.elems:] = rewards[:i]
                    self.next_states[self.elems:] = next_states[:i]
                    self.terminals[self.elems:] = terminals[:i]
                    self.weights[self.elems :] = weights[:i]

                    w = 1 / self.weights
                    w = (w) .pow(self.prioritization_drop)
                    idx_replace = list(torch.utils.data.WeightedRandomSampler(w, n-i, replacement=False))

                    if torch.any(self.rewards[idx_replace]==1):
                        #print(torch.sum(self.rewards[idx_replace]==1))
                        pass

                    self.states[idx_replace] = states[i:]
                    self.actions[idx_replace] = actions[i:]
                    self.action_probs[idx_replace] = action_probs[i:]
                    self.rewards[idx_replace] = rewards[i:]
                    self.next_states[idx_replace] = next_states[i:]
                    self.terminals[idx_replace] = terminals[i:]
                    self.weights[idx_replace] = weights[i:]

        else:

            if self.idx + n <= self.size:
                self.states[self.idx : self.idx+n] = states
                self.actions[self.idx : self.idx+n] = actions
                self.action_probs[self.idx : self.idx+n] = action_probs
                self.rewards[self.idx : self.idx+n] = rewards
                self.next_states[self.idx : self.idx+n] = next_states
                self.terminals[self.idx : self.idx+n] = terminals
                self.weights[self.idx : self.idx+n] = weights
            else:

                i = self.size-self.idx

                self.states[self.idx:] = states[:i]
                self.actions[self.idx:] = actions[:i]
                self.action_probs[self.idx:] = action_probs[:i]
                self.rewards[self.idx:] = rewards[:i]
                self.next_states[self.idx:] = next_states[:i]
                self.terminals[self.idx:] = terminals[:i]
                self.weights[self.idx :] = weights[:i]

                self.states[:n-i] = states[i:]
                self.actions[:n-i] = actions[i:]
                self.action_probs[:n-i] = action_probs[i:]
                self.rewards[:n-i] = rewards[i:]
                self.next_states[:n-i] = next_states[i:]
                self.terminals[:n-i] = terminals[i:]
                self.weights[:n-i] = weights[i:]

        self.idx = (self.idx + n) % self.size

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
        action_probs = self.action_probs[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]

        return states, actions, action_probs, rewards, next_states, terminals
    
    @torch.no_grad()
    def get_batch(self, repeat=True):

        n = self.batch_size if repeat else min(self.batch_size,self.elems)
        # w = torch.exp(self.weights[:self.elems]-self.weights[:self.elems].max())
        # w = w/w.sum() * self.prioritization + torch.full_like(self.weights[:self.elems],(1-self.prioritization)/self.elems)
        w = self.weights.pow(self.prioritization)
        self.last_idx = torch.tensor(list(torch.utils.data.WeightedRandomSampler(w, n, replacement=repeat)), device=self.device)

        states = self.states[self.last_idx] 
        actions = self.actions[self.last_idx]
        action_probs = self.action_probs[self.last_idx]
        rewards = self.rewards[self.last_idx]
        next_states = self.next_states[self.last_idx]
        terminals = self.terminals[self.last_idx]
        weights = w[self.last_idx]

        return states, actions, action_probs, rewards, next_states, terminals, weights
    
    @torch.no_grad()
    def update_weights(self, weights):

        assert self.last_idx is not None
        l=0.9
        self.weights[self.last_idx] = self.weights[self.last_idx]*l + weights*(1-l)



class Replay_Buffer_Segments():
    def __init__(   self, 
                    state_shape, 
                    action_shape,
                    params_shape,
                    segment_lenght, 
                    num_simulations,
                    num_steps = 1024, 
                    batch_size=256, 
                    device='cpu'):
        
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.states = torch.zeros((num_simulations, num_steps) + state_shape, device=device)
        self.actions = torch.zeros((num_simulations, num_steps) + action_shape, device=device)
        self.act_params = torch.zeros((num_simulations, num_steps) + params_shape, device=device)
        self.action_probs = torch.zeros((num_simulations, num_steps), device=device)
        self.rewards = torch.zeros((num_simulations, num_steps), device=device)
        self.terminals = torch.zeros((num_simulations, num_steps), dtype=torch.bool, device=device)

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.elems = 0
        self.idx = 0
        self.batch_size = batch_size
        self.segment_lenght = segment_lenght

        self.device = device

        self.last_idx = None


    @torch.no_grad()
    def add_experience(self, states, actions, action_params, action_probs, rewards, terminals):

        assert states.shape[0]==self.num_simulations, "num simulations don't match experience passed"

        self.states[:, self.idx] = states
        self.actions[:, self.idx] = actions.reshape(states.shape[0],-1)
        self.act_params[:, self.idx] = action_params.reshape(states.shape[0],-1)
        self.action_probs[:, self.idx] = action_probs
        self.rewards[:, self.idx] = rewards
        self.terminals[:, self.idx] = terminals

        self.idx = (self.idx + 1) % self.num_steps

        if self.elems < self.num_steps:
            self.elems += 1

    @torch.no_grad()
    def get_batch(self,):

        idx = torch.randint(0, self.elems - self.segment_lenght + 1, (self.batch_size,), device=self.device) + (self.idx % self.elems)
        sim_n = torch.randint(0,self.num_simulations,(self.batch_size,), device=self.device)

        idx = (idx[:,None] + torch.arange(self.segment_lenght, device=self.device)[None]) % self.elems
        sim_n = sim_n[:,None]

        states = self.states[sim_n, idx]
        actions = self.actions[sim_n, idx]
        act_params = self.act_params[sim_n, idx]
        action_probs = self.action_probs[sim_n, idx]
        rewards = self.rewards[sim_n, idx]
        terminals = self.terminals[sim_n, idx]

        return states, actions, act_params, action_probs, rewards, terminals

    @torch.no_grad()
    def get_last_segment(self,):

        start = (self.idx - self.segment_lenght) % self.elems
        end = self.idx

        if self.idx >= self.segment_lenght:
            states = self.states[:, start:end]
            actions = self.actions[:, start:end]
            act_params = self.act_params[:, start:end]
            action_probs = self.action_probs[:, start:end]
            rewards = self.rewards[:, start:end]
            terminals = self.terminals[:, start:end]
        else:
            states = torch.concat([self.states[:, start:], self.states[:, :end]], 1)
            actions = torch.concat([self.actions[:, start:], self.actions[:, :end]], 1)
            act_params = torch.concat([self.act_params[:, start:], self.act_params[:, :end]], 1)
            action_probs = torch.concat([self.action_probs[:, start:], self.action_probs[:, :end]], 1)
            rewards = torch.concat([self.rewards[:, start:], self.rewards[:, :end]], 1)
            terminals = torch.concat([self.terminals[:, start:], self.terminals[:, :end]], 1)

        return states, actions, act_params, action_probs, rewards, terminals
