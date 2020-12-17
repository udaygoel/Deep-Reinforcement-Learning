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
OU_THETA = 0.2              # how "strongly" the system reacts to perturbations - OUNoise process
OU_SIGMA = 0.25             # the variation or the size of the noise - OUNoise process

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')



class MultiAgent:
    """Interaction between multiple agents in common environment"""
    def __init__(self, state_size, action_size, num_multi_agents, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_multi_agents = num_multi_agents
        self.seed = seed
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.agents = [Agent(self.state_size, self.action_size, self.seed,
                             num_multi_agents=self.num_multi_agents, memory=self.memory)
                       for x in range(self.num_multi_agents)]
        self.steps_count = 0


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn"""

        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # update steps_count
        self.steps_count = (self.steps_count + 1) % UPDATE_EVERY

        # Learn if enough samples are available in memory and enough step counts have been reached.
        if len(self.memory) > BATCH_SIZE and self.steps_count == 0:
            for _ in range(UPDATE_TIMES):
                for num_agent, agent in enumerate(self.agents):
                    agent.step(num_agent)


    def act(self, states, add_noise=True):
        """Agents perform actions according to their policy"""
        actions = np.zeros([self.num_multi_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index].reshape(1,-1), add_noise)

        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save_model(self):
        """Save learned model paramaters of each agent"""
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index + 1))

    def load_model(self):
        """Load learned model parameters for each agent"""
        for index, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor.pth'.format(index+1)))
            agent.critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic.pth'.format(index + 1)))

class Agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, random_seed, num_multi_agents = 1, memory = None):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimonesion of each action
            random_seed (int): random seed
            num_multi_agents (int): The total number of agents in a multiple agent problem.
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_multi_agents = num_multi_agents
        self.steps_count = 0
        
        # Actor Network (with Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network (with Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, num_multi_agents=num_multi_agents).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, num_multi_agents=num_multi_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.epsilon = EPSILON
        self.noise = OUNoise(action_size, random_seed, theta=OU_THETA, sigma=OU_SIGMA)
        
        # Replay Memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed) if memory is None else memory
        
    
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
        
    
    def step(self, num_agent):

        for _ in range(UPDATE_TIMES):
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, num_agent)
            
    
    def learn(self, experiences, gamma, num_agent):
        """Update policy and value parameters using given batch of experienced tuples
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state)           -> action
            critic_target(state, action)  -> Q-value
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float) : discount factor
            num_agent (int): The index of the agent to select the values from experiences
        ======
        """
        
        states, actions, rewards, next_states, dones = experiences

        # Select only the data in experiences corresponding to index num_agent
        agent_states = states[:, num_agent, :]
        agent_next_states = next_states[:, num_agent, :]
        agent_rewards = rewards[:, num_agent].reshape(rewards.shape[0], 1)
        agent_dones = dones[:, num_agent].reshape(dones.shape[0], 1)

        # re-arranging the data from all agents into a single vector for critic network
        critic_states = states.reshape(states.shape[0], -1)
        critic_next_states = next_states.reshape(next_states.shape[0], -1)
        critic_actions = actions.reshape(actions.shape[0], -1)


        #----------------------------update critic ------------------------------------#
        # Get predicted next-state actions and Q values from target models
        # get next actions using the target actor for all the next states (all agents)
        total_agents = next_states.shape[1]
        next_actions = [self.actor_target(next_states[:, n_agent, :]) for n_agent in range(total_agents)]
        # combine next_actions to create a single vector across all agents
        critic_next_actions = torch.cat(next_actions, dim=1).to(device)
        Q_targets_next = self.critic_target(critic_next_states, critic_next_actions)
        
        # Compute Q targets for current states
        Q_targets = agent_rewards + (gamma * Q_targets_next * (1-agent_dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(critic_states, critic_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()
        
        #----------------------------update actor ---------------------------------------#
        # Compute actor loss
        # get the predicted action using the local actor for all the states (all agents)
        actions_pred = [self.actor_local(states[:, n_agent, :]) for n_agent in range(total_agents)]
        # combine predicted actions to create a single vector across all agents
        actions_pred_ = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(critic_states, actions_pred_).mean()
        
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
    
    def __init__(self, action_size, seed, mu=0., theta=0.2, sigma=0.25):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(action_size)
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
    
    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        ======
        """

        self.memory = deque(maxlen=buffer_size)   # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""

        # Join a sequence of agent's states, next states and actions along the columns
        #state = np.concatenate(state, axis=0)
        #next_state = np.concatenate(next_state, axis=0)
        #action = np.concatenate(action, axis=0)

        """
        for s, a, r, s_next, d in zip(state, action, reward, next_state, done):
            e = self.experience(s, a, r, s_next, d)
            self.memory.append(e)
        """

        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([[e.state] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([[e.action] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([[e.next_state] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device) 
        
        return (states, actions, rewards, next_states, dones)
    
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
        
