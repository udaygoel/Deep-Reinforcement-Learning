# Navigation - Value based methods



### Introduction

This report describes the learning algorithm. It also provides the details of the implementation, along with ideas for future work.



### Environment

In this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![banana image](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.



### Learning Algorithm

The project looks at 3 different algorithms of value-based methods to train the agent. In all cases, the Q-value function approximator is implemented as a 3 layer Feed Forward Neural Network. The 3 layers include Input Layer, 1 Hidden Layer and Output Layer.  The algorithms implemented are:

1. Deep Q-Network (DQN)
2. Double Deep Q-Network (DDQN)
3. DDQN and Prioritized Experience Replay (PER)

For each algorithm, we try different combinations of hyperparameter values to identify the set of hyperparameter values that achieve the fastest training. The sizes of the Input Layer and the Hidden Layer are also hyperparameters. The code inputs these hyperparameters by reading a "hyperparameters.csv" file to a pandas dataframe and iterates through each row in the dataframe. Here is a list of the hyperparameters that can be changed:

* E: Number of episodes for training
* N: Memory size / Buffer size. Used for PER.
* k: Frequency of training. That is, the number of steps after which to train the agent.
* C: Frequency of updating the weights for Target Q Network. This is used for DDQN.
* M: Batch Size for sampling memory and training
* lr: Learning Rate
* Y: Gamma
* t: Tau for target Q network parameters updated. Used for DDQN
* fc1_units: Number of nodes in the Input Layer
* fc2_units: Number of nodes in the Hidden Layer
* a: `a` parameter for PER
* b: `b` parameter for PER
* p_err: priority error term for PER



DQN and DDQN implementations use 9 different combinations of the hyperparameter values

DDQN + PER implementation uses 5 combinations of values for (a, b, p_err) for each of the 9 hyperparameter values combinations used by DQN and DDQN. So, this implementation uses 45 different combinations in total. Since training with all 45 combinations takes a very long time, these parameters were split into 5 parts - 4 parts have 10 combinations each and the 5th part has 5 combinations. 



The results of these 63 (9 + 9 + 45) combinations are then compared in 3 steps:

1. The results of the DDQN+PER are first studied. Remember, we use 5 combinations of values for (a, b, p_err) for each of the 9 hyperparameter values combinations used by DQN and DDQN. In this step, we find the values of (a, b, p_err) that give the fastest training (the smallest episode to complete training) out of the 5 combinations. This reduces the results from a 45 row dataframe to a 9 row dataframe and allows us to compare the results with DQN and DDQN implementations. We can now check in next step the benefit PER added to the training by using the 9 row dataframe
2. We combine the results of the 3 implementations. We keep track of the algo used by using 3 boolean columns, one each for DQN("dqn"), DDQN("ddqn") and PER("er"). The columns work as below: 
   - For DQN implementation, dqn column is True and rest are false.
   - For DDQN implementation, ddqn column is True and rest are false.
   - For DDQN+PER implementation, dqn column is True and rest are false.
3. sdfs
4. sdf





* Deep Q-Network (DQN): 





