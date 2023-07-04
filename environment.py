import torch
import torch.nn.functional as F
import math

class SimulationEnvironment0:
    def __init__(self, num_simulations=1, 
                 num_blackholes=2, 
                 force_constant=0.1, 
                 velocity_scale=0.1, 
                 crash_reward=-1.0, 
                 goal_reward=1.0, 
                 goal_threshold=0.05, 
                 max_steps = 100,
                 device='cpu'):
        self.num_simulations = num_simulations
        self.num_blackholes = num_blackholes
        self.force_constant = force_constant
        self.velocity_scale = velocity_scale
        self.crash_reward = crash_reward
        self.goal_reward = goal_reward
        self.crash_threshold = math.pow(force_constant,1/3)
        self.goal_threshold = goal_threshold
        self.device = device
        self.max_steps = max_steps

        self.steps = torch.zeros(num_simulations, device=device)
        # Generate random states for space ship, goal, and black holes
        self.states = torch.rand((num_simulations, 1 + num_blackholes + 1, 2), device=device)
        # self.states[:,1] = 0.25
        # self.states[:,2:] = 0.75

    def get_state(self):
        return self.states.clone()

    def reset(self, is_terminal):
        self.states[is_terminal] = torch.rand((is_terminal.sum().item(), 1 + self.num_blackholes + 1, 2), device=self.device)
        # self.states[:,1] = 0.25
        # self.states[:,2:] = 0.75
        self.steps[is_terminal]=0

    @torch.no_grad()
    def step(self, actions):
        rewards = torch.zeros(self.num_simulations)
        #next_states = self.states.clone()

        # Update ship's position based on actions and attractive forces from black holes
        ship_positions = self.states[:, 0, :]
        goal_position = self.states[:, 1, :]
        blackhole_positions = self.states[:, 2:, :]
        goal_distance_before = torch.norm(ship_positions - goal_position, dim=1)

        distance = blackhole_positions - ship_positions.unsqueeze(1)
        inv_distance = 1 / torch.norm(distance, dim=2)
        direction = distance / torch.norm(distance, dim=2, keepdim=True)
        forces = self.force_constant * direction * inv_distance.unsqueeze(2)
        ship_velocity = self.velocity_scale * actions + forces.sum(dim=1)
        next_ship_positions = torch.clamp(ship_positions + ship_velocity, 0, 1)
        self.states[:, 0, :] = next_ship_positions

        goal_distance_after = torch.norm(next_ship_positions - goal_position, dim=1)
        rewards = goal_distance_before - goal_distance_after

        # Check for terminal conditions (crash into a black hole or reach the goal)
        distance_to_blackholes = torch.norm(next_ship_positions.unsqueeze(1) - blackhole_positions, dim=2)
        #is_crashed = (self.force_constant / distance_to_blackholes.pow(2)) > distance_to_blackholes
        is_crashed = distance_to_blackholes < self.crash_threshold
        is_crashed = is_crashed.any(dim=1)
        is_goal_reached = goal_distance_after < self.goal_threshold
        self.steps+=1
        is_terminal = is_crashed | is_goal_reached | (self.steps>=self.max_steps)

        rewards[is_crashed] = self.crash_reward
        rewards[is_goal_reached] = self.goal_reward

        # if torch.any(is_terminal):
        #     #print(rewards[is_terminal])
        #     print(self.states[is_terminal,0])

        # Reset terminated simulations
        if torch.any(is_terminal):
            self.reset(is_terminal)

        # if torch.any(is_terminal):
        #     #print(rewards[is_terminal])
        #     print(self.states[is_terminal,0])

        return rewards, self.states.clone(), is_terminal
    







"""
from environment import SimulationEnvironment0

sim = SimulationEnvironment0(num_blackholes=2,force_constant=0.002,velocity_scale=0.01)
g_radious = sim.goal_threshold
bh_radious = sim.crash_threshold

from IPython.display import clear_output

import matplotlib.collections as mc

for i in range(10):
    actions = torch.rand((1,2))*2-1
    rewards, next_states, is_terminal = sim.step(actions,)
    print(rewards, next_states, is_terminal)
    states = sim.get_state()
    print(states)
    fig, ax = plt.subplots()
    plt.plot(states[0,0,0],states[0,0,1],'b*',markersize=10)      # ship
    #circles = plt.Circle(states[0,2], 0.2, color='r')
    p_bh = [plt.Circle(state, bh_radious) for state in states[0,2:]]
    c_bh = mc.PatchCollection(p_bh, color='red')
    ax.add_collection(c_bh)
    p_g = plt.Circle(states[0,1], g_radious, color='green')
    ax.add_artist(p_g)
    ax.set_aspect('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    clear_output(wait=True)
    plt.show()
    time.sleep(0.01)
"""