import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
FC1_UNITS = 128         # Size of the first hidden layer
FC2_UNITS = 64          # Size of the second hidden layer
PRIORITY_ERR = 0.001    # Error for the priority replay probability
PRIORITY_A = 0.5        # Hyperparameter a for experience replay
PRIORITY_B = 0.5        # Hyperparameter b for experience replay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hyperp = None, ddqn=False, 
                 hyperp_exp_replay = None, exp_replay=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Load hyperparameters if not None
        global BUFFER_SIZE, UPDATE_EVERY, BATCH_SIZE, LR, GAMMA, TAU, FC1_UNITS, FC2_UNITS
        global PRIORITY_ERR, PRIORITY_A, PRIORITY_B
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.exp_replay = exp_replay
   
        
        if hyperp is not None:
            N, k, C, M, lr, Y, t, fc1, fc2 = hyperp
            BUFFER_SIZE = int(N)
            UPDATE_EVERY = int(C)
            BATCH_SIZE = int(M)
            LR = lr
            GAMMA = Y
            TAU = t
            FC1_UNITS = int(fc1)
            FC2_UNITS = int(fc2)
            
        if hyperp_exp_replay is not None:
            a, b, err = hyperp_exp_replay
            PRIORITY_A = a
            PRIORITY_B = b
            PRIORITY_ERR = err
        

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units=FC1_UNITS,
                                      fc2_units=FC2_UNITS).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units=FC1_UNITS, 
                                       fc2_units=FC2_UNITS).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # calculate priority for this step
        if self.exp_replay:
            
            pr = {
                'state': torch.from_numpy(state).float().unsqueeze(0).to(device),
                'action': torch.from_numpy(np.vstack([action])).long().to(device),
                'reward': torch.from_numpy(np.vstack([reward])).float().to(device),
                'next_state': torch.from_numpy(next_state).float().unsqueeze(0).to(device),
                'done': torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)
            }

            qtarget, qestimate = self.qvalues(pr['state'], pr['action'], pr['reward'], pr['next_state'], pr['done'])
            priority = np.power(np.abs((qtarget - qestimate).data[0][0].to('cpu').numpy()) + PRIORITY_ERR, PRIORITY_A)
        
        else:
            priority = 1.0
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(self.exp_replay)
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, priority) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, p_priorities = experiences

        ## compute and minimize the loss
        
        # get the target q-value and estimated q-value        
        qtarget, qestimate = self.qvalues(states, actions, rewards, next_states, dones)
        
        # Compute loss. First get importance sampling weight for prioritized experience replay
        # The sampling weight does not need gradient descent, so switching this off first
        with torch.no_grad():
            if self.exp_replay:
                imp_sampling_wgt = torch.pow(len(self.memory.memory)*p_priorities, -PRIORITY_B).data
            else:
                imp_sampling_wgt = 1.0

        qtarget = qtarget * imp_sampling_wgt
        qestimate = qestimate * imp_sampling_wgt
         
        loss = F.mse_loss(qestimate, qtarget) 
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update prioritized experience replay priorities
        if self.exp_replay:
            td_error = (qtarget - qestimate).data.to('cpu').numpy()
            self.memory.update_priority(td_error)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)         
        
    def qvalues(self, states, actions, rewards, next_states, dones):
        # Function to return the TD target and TD old model values
        # These values can be used to calculate the priority for experience replay and run the learning
        # algo
        
        # get the target value. Double DQN algorithm applied if selected
        if self.ddqn:
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            qtarget_next_state = self.qnetwork_target(next_states).detach().gather(1,best_actions)
        else:
            qtarget_next_state = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        qtarget = rewards + (GAMMA * qtarget_next_state*(1-dones))
        
        # get the old, estimated value
        qestimate = self.qnetwork_local(states).gather(1, actions)
        
        return qtarget, qestimate
    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.memory_list = list(self.memory)
        
        # priorities for each item in memory. Set up as numpy array, instead of adding to memory for fast computation of priority 
        # probabilities and quick update to the values on updating the Q network
        self.priorities = np.zeros(buffer_size) 
        self.priorities_sum = 0
        
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.experiences_idx = None
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        
        # update priorities by shifting to left
        if len(self.memory) < len(self.priorities):
            # add more values to array as full length not filled yet
            self.priorities[len(self.memory)] = priority
            self.priorities_sum += priority
        else:
            # shift values to left like a deque
            self.priorities_sum += ( -self.priorities[0] + priority)
            self.priorities[:-1] = self.priorities[1:]
            self.priorities[-1] = priority

            
        self.memory.append(e)
    
    def sample(self, exp_replay=False):
        """Randomly sample a batch of experiences from memory."""
        if exp_replay:
            # Get the prioirities probability for each experience in memory
            p_priorities = self.priorities / self.priorities_sum
           
            self.experiences_idx = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, 
                                                    replace=False, p=p_priorities[:len(self.memory)])
            experiences = [self.memory[i] for i in self.experiences_idx]
            
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            priorities_sum = np.array(1.0)


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        p_priorities = torch.from_numpy(np.vstack(self.priorities[self.experiences_idx])).float().to(device)
        p_priorities = p_priorities.div(self.priorities_sum)
        
        return (states, actions, rewards, next_states, dones, p_priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_priority(self, td_error):
        # update the priority value in the memory for the experiences picked for learning
        for i,idx in enumerate(self.experiences_idx):
            pr = np.power(np.abs(td_error[i][0]) + PRIORITY_ERR, PRIORITY_A)
            self.priorities_sum += (-self.priorities[idx] + pr)
            self.priorities[idx] = pr
            
        
        return
        