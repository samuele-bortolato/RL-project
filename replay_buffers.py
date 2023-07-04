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
                    prioritization=0.9,
                    prioritized_drop = False, 
                    prioritization_drop=0.9, 
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
                    w = (w) * self.prioritization_drop + torch.full_like(self.weights[:self.elems], (1-self.prioritization_drop)/self.elems*w.sum())
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
        w = torch.exp(self.weights[:self.elems]-self.weights[:self.elems].max())
        w = w/w.sum() * self.prioritization + torch.full_like(self.weights[:self.elems],(1-self.prioritization)/self.elems) 
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



    
