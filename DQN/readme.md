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
  

![continous-DQN for Pendulum-v0]https://github.com/SchindlerLiang/Deep-Reinforcement-Learning/blob/master/DQN/img/continous_DQN_structure.png


![continous-DQN for Pendulum-v0]https://github.com/SchindlerLiang/Deep-Reinforcement-Learning/blob/master/DQN/img/Pendulum-v0_DQN_result.png
