# Navigation - Value based methods


### Introduction

In this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![banana image](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

Follow the instructions below to set up and explore the environment.

1. Follow the instruction in the [Udacity's DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). The instructions can be found in README.md at the root of that repository. This will install PyTorch, the ML-Agents toolkit and a few more Python packages required to run the script in this repository.
    
    (_For Windows Users_) The ML-Agents toolkit supports Windows10. While it might be possible to run the ML0Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.
    
2. Download the environment from one of the links below. The environment has already been built and you need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

### Instructions

Once the environment has been set up, follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

The details of the implementation and results can be found in `Report.md` . 



### Project Files

The project uses these files and folders:

* Report.md: This report describes the learning algorithm. It provides the details of the implementation, along with ideas for future work.
* [Navigation.ipynb]() : This Jupyter Notebook covers the project. It starts with the analysis of the environment and then trains the agent to solve the environment. There are multiple algorithms used for training and their performance is compared at the end of training. Finally the performance of the trained agent is observed by solving the environment over 100 episodes.
* dqn_agent.py: script to create the Agent class. This class includes functions for training the agent, as well as providing an action given an environment state.
* model.py: This script defines the deep neural network used to estimate the Q-values for each action, given an environment state.
* checkpoint_ddqn_er_4.pth: The checkpoint file with the saved state dictionary values of the trained agent.
* checkpoint folder: This folder contains the checkpoint files for the different hyperparameter values
* dqn_solved: csv and pickle file with the results from different hyperparameter values using the Deep Q-Learning Network (DQN) for Q-value approximation
* ddqn_solved: csv and pickle file with the results from different hyperparameter values using the Double Deep Q-Learning Network (DDQN) algorithm 
* ddqn_er_solved_i: Five csv and pickle files with the results from different hyperparameter values using the Double Deep Q-Learning Network (DDQN) and Prioritized Experience Replay (ER) algorithms. "i" ranges from 1 to 5. Due to the large number of iterations, the hyperparameters were broken into 5 sets and the script was run on each set separately. The files store the results for each set respectively.
*  hyperparameters_dqn_ddqn.csv: The table of different hyperparameter values used for training with DQN and DDQN algos.
* hyperparameters: The last table of hyperparameter values used for training with DDQN and Prioritized Experience Replay
* hyperparameters testing results.xlsx: This file pulls together the results from all the different algos and hyperparameter values iterations and compares them in a tabular form. It also gives an index to explain what each hyperparameter symbol means. 













 


