import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 128           # minibatch size
GAMMA = .99                # discount factor
TAU = 1e-3                 # for soft update of target parameters
LR_ACTOR = 1e-4            # learning rate of the actor
LR_CRITIC = 1e-3           # learning rate of the critic
WEIGHT_DECAY = 0           # L2 weight decay
UPDATE_EVERY = 10           # How often to update the network
UPDATE_TIMES = 10          # How many times to update the network each time

EPSILON = 1.0              # epsilon for the noise process added to the actions
EPSILON_DECAY = 1e-6       # decay for epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class Agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, random_seed, num_agents=1):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimonesion of each action
            random_seed (int): random seed
            num_agents (int): number of agents to create in the same object. All agents share same network weights.
            
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.steps_count = 0
        
        # Actor Network (with Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network (with Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.epsilon = EPSILON
        self.noise = OUNoise(action_size, random_seed, num_agents)
        
        # Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy"""
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample() * np.maximum(self.epsilon, 0.2)
        self.epsilon -= EPSILON_DECAY
        
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn"""
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        # update steps_count
        self.steps_count = (self.steps_count + 1) % UPDATE_EVERY
        
        # Learn, if enough samples are available in memory and enough step counts have been reached.
        if len(self.memory) > BATCH_SIZE and self.steps_count == 0:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            #self.noise.reset()           # resetting noise
            
    
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experienced tuples
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state)           -> action
            critic_target(state, action)  -> Q-value
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float) : discount factor
        ======
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        #----------------------------update critic ------------------------------------#
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()
        
        #----------------------------update actor ---------------------------------------#
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #----------------------------update target networks -----------------------------#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters
        theta_target = tau * theta_local + (1 - tau) * theta_target
        
        Params
        ======
            local_model:  PyTorch model (weights will be copied from)
            target_model: Pytorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        
        
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, action_size, seed, num_agents, mu=0., theta=0.2, sigma=0.25):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((num_agents, action_size))
        #self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        """Reset the internal state (=noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.standard_normal(size=x.shape))
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
    
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        ======
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)   # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        for s, a, r, s_next, d in zip(state, action, reward, next_state, done):
            e = self.experience(s, a, r, s_next, d)
            self.memory.append(e)
        """       
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        """
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device) 
        
        return (states, actions, rewards, next_states, dones)
    
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
        
