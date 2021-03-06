from Network import FeedForwardNN
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np
import cv2

class PPO:
    def __init__(self, env):
        #Initialise hyperparameters
        self._init_hyperparameters()
        
        #Extract environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        #ALG STEP 1
        #Initialise actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        #Variable for covariance matrix
        self.cov_var = torch.full(size = (self.act_dim,), fill_value = 0.5)
        
        #Create covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
        
        #Initialise Adam optimiser for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
    def get_action(self, obs):
        #Get mean from actor (Identical to self.actor.forward(obs))
        mean = self.actor(obs)
        
        #Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        
        #Sample action from the distribution
        action = dist.sample()
        
        #Get log prob from action from distribution
        log_prob = dist.log_prob(action)
        
        #Isolate tensors from graphs and return action and log_prob
        return action.detach().numpy(), log_prob.detach()
        
    def compute_rtgs(self, batch_rews):
        #Rewards-to-go per episode per batch to return
        #Shape will be (number of timesteps per episode)
        batch_rtgs = []
        
        #Interate through episodes backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 #Discounted reward so far
            
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        #Convert rewards-to-go to tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype = torch.float)
        
        return batch_rtgs
        
    def _init_hyperparameters(self):
        self.max_timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.epochs = 5
        self.clip = 0.2 #Recommended value from PPO original paper
        self.lr = 0.005
        
    def learn(self, t_max):
        #Timesteps simulated so far
        t_now = 0
        
        #ALG STEP 2
        while t_now < t_max:
            
            if self.env.game_done:
                self.env.reset()
            
            #ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            #Increment current timesteps by timesteps gained this batch
            t_now += np.sum(batch_lens)
            
            #Calculate V_(phi, k)
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            #ALG STEP 5
            #Calculate advantage
            A_k = batch_rtgs - V.detach()
            
            #Normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) #add 1e-10 to avoid divide-by-zero
            
            for _ in range(self.epochs): #ALG STEP 6 & 7
                #Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                #Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                #Calculate surrogate losses
                surr1 = ratios * A_k #Raw ratios
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k #Binds ratios within clip hyperparameter from 1

                #Negative minimum of losses causes Adam optimiser to maximise loss
                #Then get single loss by getting the mean
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                #Calculate critic loss as mean-squared error of predictions and rewards-to-go
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                #Calculate gradients and perform backpropagation on actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()
                
                #Calculate gradients and perform backpropagation on critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
    def evaluate(self, batch_obs, batch_acts):
        #Get predictions V for each obs in batch_obs from critic network, squeezing to reduce tensor dimensions to 1
        V = self.critic(batch_obs).squeeze()
        
        #Calculate log probabilites of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        #Return predictions V and log probs log_probs
        return V, log_probs
            
    def rollout(self):
        #Batch data
        batch_obs = [] #observations
        batch_acts = [] #actions
        batch_log_probs = [] #log probabilities of each action
        batch_rews = [] #rewards
        batch_rtgs = [] #rewards-to-go
        batch_lens = [] #episodic lengths in batch
        
        #Timesteps simulated so far
        t_now = 0
        
        while not self.env.game_done:
        #while t_now < self.max_timesteps_per_batch:
            #Rewards this episode
            ep_rews = []
            
            data = self.env.emu.step([])
            obs = data["frame"]
            # obs = cv2.resize(obs, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            obs = np.expand_dims(obs, 0)
            print(obs.shape)
            #obs = self.env.reset()
            done = False
            
            while not self.env.stage_done:
                while not self.env.round_done:
            #for ep_t in range(self.max_timesteps_per_episode):
                #Increment timesteps ran this batch so far
                    t_now += 1

                    #Collect observation
                    batch_obs.append(obs)

                    action, log_prob = self.get_action(obs)
                    obs, rew, round_done, stage_done, game_done = self.env.step(action)

                    #Collect reward, action and log prob
                    ep_rews.append(rew)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)

                    if done:
                        break
                    
            #Collect episode length and reward
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        #Reshape data as tensors for drawing computation graphs
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        #ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        #Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
