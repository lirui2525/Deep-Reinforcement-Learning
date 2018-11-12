"""
DDPG ---- Gaussian process version

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.0005    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.9     # reward discount
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
A_BOUND = 2
RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.hidden_units = 50
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        
        self.state_ph = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.r_ph = tf.placeholder(tf.float32, [None, 1], 'r')
        self.next_q_ph = tf.placeholder(tf.float32, [None, 1], 'next_q')
        
        self.a,self.ae_params = self._build_actor_net(self.state_ph, scope='Actor')
        self.q,self.ce_params = self._build_critic_net(self.state_ph, self.a, scope='Critic')

        with tf.variable_scope('c_train'):
            q_target = self.r_ph + GAMMA * self.next_q_ph
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error,var_list=self.ce_params)

        with tf.variable_scope('a_train'):
            a_loss = - tf.reduce_mean(self.q)    # maximize the q
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss,var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def get_q(self,s):
        if s.ndim<2: s=s[np.newaxis, :]
        return self.sess.run(self.q,feed_dict={self.state_ph: s})

    def choose_action(self, s):
        if s.ndim<2: s=s[np.newaxis, :]
        return self.sess.run(self.a, {self.state_ph: s})[0]

    def learn(self):

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        next_q = self.get_q(bs_)
        self.sess.run(self.atrain, {self.state_ph: bs})
        self.sess.run(self.ctrain, {self.state_ph: bs, self.a: ba, self.r_ph: br, self.next_q_ph:next_q})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  
        self.memory[index, :] = transition
        self.pointer += 1



    def _build_actor_net(self, s, scope, trainable=True):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, self.hidden_units, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            action = tf.multiply(a, self.a_bound, name='scaled_a')
        a_params =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return action,a_params

    def _build_critic_net(self, s, a, scope, trainable=True):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.s_dim, self.hidden_units], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, self.hidden_units], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hidden_units], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            q =  tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
        q_params =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return q,q_params
###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # 
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -A_BOUND, A_BOUND)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)
        var *= .9999
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('ROUND:', i, ' SCORES: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break
