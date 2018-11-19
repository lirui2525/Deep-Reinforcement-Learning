Traditional implementation of Deep Q Network with experience replay(DQN.py):
  1. This implementation can only be adopted by cases with discrete action spaces.
  2. Deep Q Network Structure Q(s):
    1): input layer: state (s)
    2): output layer: n-action (n_regression)
  3. get q value for state s:  return max( Q(s) )
  4. choose action for state s: return a[ argmax( Q(s) )  ]
  5. disadvantages:
     1): can only be used for discrete action spaces
     2): NOT EASILY adopted by cases with multi-dimensional actions
  

Continous Deep Q Network with experience replay(DQN_continous.py):
  1. A modified version for DQN, which can be used for multi-dimensional continous action spaces;
  2. The action-choosing process the get_q(s) function is modified by using searching-based optimization method--Particle Swarm Optimization(PSO);

The network structure is:
![Netowkr structure for continous-DQN](https://github.com/SchindlerLiang/Deep-Reinforcement-Learning/blob/master/DQN/img/continous_DQN_structure.png)
The continous_DQN can be EASILY adopted by multi-dimensional cases,since actions are treated as n-dimension input.


PSO introduces two main hyper-parameters (swarmsize and maxiter). Large swarmsize or maxiter may cause unacceptable training time for DQN, which is the main disadvantage.

The following presents the results of continous_DQN on Pendulum-v0:
![continous-DQN for Pendulum-v0](https://github.com/SchindlerLiang/Deep-Reinforcement-Learning/blob/master/DQN/img/Pendulum-v0_DQN_result.png)


