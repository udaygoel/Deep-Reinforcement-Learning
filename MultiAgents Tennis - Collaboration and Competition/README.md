[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://video.udacity-data.com/topher/2018/August/5b81cd05_soccer/soccer.png "Soccer"


# MultiAgents Tennis - Collaboration and Competition

### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment and __Multi-Agent Deep Deterministic Policy Gradients (MADDPG)__ algorithm.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

Follow the instructions below to set up and explore the environment. 

1. Activate the Environment

Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

2. Download the Unity Environment.

   Download the environment from one of the links below. The environment has already been built and you need only select the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): 

3. Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

   (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (*To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)

__*Note*:__ The project environment is similar to, but __not identical to the Tennis environment__ on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

### Instructions

Once the environment has been set up, follow the instructions in `Tennis.ipynb` to get started with training your own agent!

The details of the implementation and results can be found in `Report.md`.



 ### Project Files

The project uses these files:

* [Report.md](https://github.com/udaygoel/Deep-Reinforcement-Learning/blob/master/Continuous%20Control%20-%20Actor%20Critic%20Methods/Report.md): This report describes the learning algorithm. It provides the details of the implementation, along with ideas for future work.
* [Tennis.ipynb](https://github.com/udaygoel/Deep-Reinforcement-Learning/blob/master/MultiAgents%20Tennis%20-%20Collaboration%20and%20Competition/Tennis.ipynb): This Jupyter Notebook covers the project. It starts with the analysis of the environment and then trains the agent to solve the environment. The performance of the trained agent is observed by solving the environment over 100 episodes. The implementations builds on the DDPG algorithm implementation in [Continuous Control - Actor Critic Methods](https://github.com/udaygoel/Deep-Reinforcement-Learning/tree/master/Continuous%20Control%20-%20Actor%20Critic%20Methods) project and adds additional features to enable it to work for multiple agents. Further details about them can be found in the `Report.md` file.
*  `maddpg_agent.py`: script to create the MultiAgent class. This script also includes the implementation for individual agents through the Agent class which covers both the actor and the critic. This class includes functions for training the agent, as well as providing an action given an environment state.
* `model.py`: This script defines the deep neural network used for implementing the Actor and Critic networks. 
* `agent1_checkpoint_actor.pth`: Saved trained actor network weights of the successful first agent
* `agent1_checkpoint_critic.pth`: Saved trained critic network weights of the successful first agent
* `agent2_checkpoint_actor.pth`: Saved trained actor network weights of the successful second agent
* `agent2_checkpoint_critic.pth`: Saved trained critic network weights of the successful second agent
* `workspace_utils.py`: script provided by Udacity 

### (Optional) Challenge: Play Soccer

After you have successfully completed the project, you might like to solve a more difficult environment, where the goal is to train a small team of agents to play soccer.

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

![Soccer][image2]

To solve this harder task, you'll need to download a new Unity environment. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

After you have followed the instructions above, open `Soccer.ipynb` (located in the `p3_collab-compet/` folder in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning)) and follow the instructions to learn how to use the Python API to control the agent.

(*For AWS*) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents. (*To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)

