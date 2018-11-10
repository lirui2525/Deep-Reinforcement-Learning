# A simple version of PPO adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py

# view more in Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]


import tensorflow as tf
import numpy as np
import gym
import random


EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
CLIP_VALUE = 0.2
CAPACITY = 64
A_BOUND = 2
HIDDEN_UNITS = 100


class PPO:
    def __init__(self):
        self.sess = tf.Session()
        self.hidden = HIDDEN_UNITS
        self.state_dim = S_DIM
        self.action_dim = A_DIM
        self.pointer = 0
        self.reset_data()
        self.a_bound = A_BOUND
        
        self.state_ph = tf.placeholder(tf.float32,[None,self.state_dim],'state')
        self.disc_r_ph = tf.placeholder(tf.float32,[None,1],'dis_r')
        
        self.q, q_paras = self._build_critic_net('critic',trainable=True)
        
        self.adv_ph = tf.placeholder(tf.float32,[None,1],'advantage')
        self.action_ph = tf.placeholder(tf.float32,[None,self.action_dim],'action')

        train_pi,pi_paras = self._build_actor_net('actor_train',trainable=True)
        old_pi,oldpi_paras = self._build_actor_net('actor_old',trainable=False)


        with tf.variable_scope('closs'):
            self.advantage = self.disc_r_ph - self.q
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.c_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        with tf.variable_scope('aloss'):
            ratio = train_pi.prob(self.action_ph) / old_pi.prob(self.action_ph)
            surrogate_loss = tf.clip_by_value(ratio,1.0-CLIP_VALUE,1.0+CLIP_VALUE) * self.adv_ph
            self.aloss = -tf.reduce_mean( tf.minimum(surrogate_loss,ratio*self.adv_ph) )
            self.a_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            self.a_update_op = [ old_para.assign(para) for para,old_para in zip(pi_paras,oldpi_paras)]
            self.a_sample_op = tf.squeeze(old_pi.sample(1),axis=0)
        
        self.sess.run(tf.global_variables_initializer())
        
        
    def choose_action(self,s):
        if s.ndim<2: s =s[np.newaxis,:]
        a =  self.sess.run(self.a_sample_op,feed_dict={
                self.state_ph:s
                })[0]
        return np.clip(a, -self.a_bound, self.a_bound)
            
            
    def update(self, s, a, r):
        self.sess.run(self.a_update_op)
        adv = self.sess.run(self.advantage, {self.state_ph: s, self.disc_r_ph: r})

        # update actor
        [self.sess.run(self.a_train_op, {self.state_ph: s, self.action_ph: a, self.adv_ph: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.c_train_op, {self.state_ph: s, self.disc_r_ph: r}) for _ in range(C_UPDATE_STEPS)]

            
            
    def get_v(self,s):
        if s.ndim<2: s=s[np.newaxis,:]
        return self.sess.run(self.q,feed_dict={
               self.state_ph:s
               })[0]

    
    def reset_data(self):
        self.data = {'s':[],'a':[],'r':[],'dis_r':[]}

    def process_data(self,s,a,r,s_):
        self.pointer += 1
        self.data['s'].append(s)
        self.data['a'].append(a)
        self.data['r'].append(r)
        q_value = self.get_v(s_)
        self.data['dis_r'] = []
        for single_r in self.data['r'][::-1]:
            q_value = single_r + GAMMA * q_value
            self.data['dis_r'].append(q_value)
        self.data['dis_r'].reverse()
        
        if self.pointer % CAPACITY == 0:
            chosen_index = random.sample(range( len(self.data['s']) ), BATCH   )
            tr_s = np.vstack(np.array(self.data['s'])[chosen_index])
            tr_a = np.vstack(np.array(self.data['a'])[chosen_index])
            tr_dis_r = np.vstack(np.array(self.data['dis_r'])[chosen_index])
            self.update(tr_s,tr_a,tr_dis_r)
            self.reset_data()

    
    def _build_critic_net(self,scope_name,trainable):
        with tf.variable_scope(scope_name):
            h1 = tf.layers.dense(self.state_ph,units=self.hidden,activation=tf.nn.relu,trainable=trainable)
            q =  tf.layers.dense(h1,units=1,trainable=trainable)
        q_paras = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        return q, q_paras
        
        
    def _build_actor_net(self,scope_name,trainable):
        with tf.variable_scope(scope_name):
            h1 = tf.layers.dense(self.state_ph,units=self.hidden,activation=tf.nn.relu,trainable=trainable)
            mu = self.a_bound * tf.layers.dense(h1,units=self.action_dim,activation=tf.nn.tanh,trainable=trainable)
            sigma = tf.layers.dense(h1,units=self.action_dim,activation=tf.nn.softplus,trainable=trainable)
            action = tf.distributions.Normal(loc=mu,scale=sigma)
        a_paras = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        return action, a_paras
   
        
env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)        
        ppo.process_data(s,a, [(r+8)/8] ,s_)
        s = s_
        ep_r += r

        if t == EP_LEN - 1:
            print('round ' +  str(ep) + ' scores ' + str(ep_r))
        
        
