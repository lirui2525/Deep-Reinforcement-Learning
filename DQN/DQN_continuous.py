"""
DDPG ---- Ornstein Uhlenbeck noise function 

Ornstein Uhlenbeck noise function is introduced from https://github.com/liampetti/DDPG

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
from pyswarm import pso
import matplotlib.pyplot as plt
#####################  hyper parameters  ####################

MAX_EPISODES = 150
MAX_EP_STEPS = 200
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.9     # reward discount
MEMORY_CAPACITY = 1000
BATCH_SIZE = 16
RENDER = False
ENV_NAME = 'Pendulum-v0'
EPSILON = 0.99
EPSILON_MIN = 0.03
DECAY_RATIO = 0.95
A_BOUND = 2
###############################  DDPG  ####################################

class DQN(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.hidden_units = 50
        self.a_dim, self.s_dim = a_dim, s_dim
        
        self.a_bound = A_BOUND
        
        self.state_ph = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.action_ph = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.disc_r_ph = tf.placeholder(tf.float32, [None, 1], 'disc_r')
        

        
        self.q,self.q_params = self._build_q_net(self.state_ph, self.action_ph, scope='Critic')

        with tf.variable_scope('c_train'):
            c_loss = tf.reduce_mean( tf.squared_difference(self.disc_r_ph,self.q)  )
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(c_loss,var_list=self.q_params)

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def get_q(self,s):
        if s.ndim<2: s = s[np.newaxis,:]
        next_action = self.choose_action(s).reshape(1,-1)        
        next_q_value = self.sess.run(self.q,feed_dict={self.state_ph: s,self.action_ph:next_action})
        return next_q_value

    def choose_action(self, s):
        if s.ndim<2: s = s[np.newaxis,:]
        def q_function(ind):
            if np.array(ind).ndim<2: ind = np.array(ind)[np.newaxis,:]
            return -float(self.sess.run(self.q,feed_dict={self.state_ph: s,self.action_ph:ind})[0] )       
        a_opt, q_opt = pso(q_function, lb=[-self.a_bound], ub=[self.a_bound],swarmsize=5,maxiter=2)
        if np.array(a_opt).ndim<2: a_opt = np.array(a_opt)[np.newaxis,:]
        return a_opt
    
    
    def learn(self):

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        next_q = np.vstack( [self.get_q(single_s) for single_s in bs_] )
        dis_r = br + GAMMA * next_q
        self.sess.run(self.ctrain, {self.state_ph: bs,self.disc_r_ph:dis_r,self.action_ph:ba})

    def store_transition(self, s, a, r, s_):
        if s.ndim<2: s = s[np.newaxis,:]
        if a.ndim<2: a = a[np.newaxis,:]
        if r.ndim<2: r = r[np.newaxis,:]
        if s_.ndim<2: s_ = s_[np.newaxis,:]
        
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_q_net(self, s, a, scope, trainable=True):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.s_dim, self.hidden_units], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, self.hidden_units], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hidden_units], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net1,self.hidden_units, trainable=trainable)            
            q =  tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
        q_params =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return q,q_params
    
###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = 1

dqn = DQN(a_dim, s_dim)


result = []
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        EPSILON = EPSILON * DECAY_RATIO if EPSILON > EPSILON_MIN else EPSILON_MIN
        
        #env.render()

        if np.random.random() < EPSILON:
            a = np.random.uniform(-A_BOUND,A_BOUND,(1))
        else:
            a = dqn.choose_action(s)
        s_, r, done, info = env.step( a)
        s_ = s_.reshape(1,-1)
        dqn.store_transition(s, a, np.array([r])/10, s_)
        if dqn.pointer > MEMORY_CAPACITY:
            dqn.learn()

        s = s_
        ep_reward += r
        if done or j == MAX_EP_STEPS-1:
            print('ROUND:', i, ' SCORES: %i' % int(ep_reward) )
            if ep_reward > -300:RENDER = True
            result.append(ep_reward)
            break
plt.plot(result)