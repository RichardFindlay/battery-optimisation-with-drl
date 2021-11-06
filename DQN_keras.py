import gym
from gym import error, spaces, utils
import floris.tools as wfct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, subprocess, time, signal
import random
import logging
from datetime import datetime
from collections import deque

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam




#make a DQN object

class DQN:
	def __init__(self, env):
		self.env = env
		self.memory = deque(maxlen=2000)
		self.model_weights = "model_weights.h5"

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.tau = 0.125

		self.model = self.create_model()
		self.target_model = self.create_model()

	def create_model(self):

		if os.path.isfile(self.model_weights):
			self.epilson = self.epsilon_min
			return model.load_weights(self.model_weights)

		model = Sequential()
		state_size = self.env.observation_space.shape
		action_size = self.env.action_space.n


		optimizer = Adam(lr=self.learning_rate)
		model.add(Dense(24, input_dim=state_size[0], activation='relu'))
		model.add(Dense(48, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(action_size, activation='linear'))
		model.compile(loss="mean_squared_error", optimizer=optimizer)
		return model


	def action(self, state, epsilon):
		# self.epsilon *= self.epsilon_decay
		# self.epsilon = max(self.epsilon_min, self.epsilon)
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		# print(self.epsilon)
		return np.argmax(self.model.predict(state)[0])


	def store(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])


	def replay(self):
		batch_size = 32
		x_batch, y_batch = [], []

		if len(self.memory) < batch_size:
			return

		samples = random.sample(self.memory, batch_size)

		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.target_model.predict(state)

			if done:
				target[0][action] = reward
			else:
				Q_future = np.max(self.target_model.predict(new_state)[0])
				target[0][action] = reward + Q_future * self.gamma
			x_batch.append(state[0])
			y_batch.append(target[0])

		
		self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=1, verbose=0)



	def target_train(self):
		w = self.model.get_weights()
		tar_w = self.target_model.get_weights()

		for i in range(len(tar_w)):
			tar_w[i] = w[i] * self.tau + tar_w[i] * (1 - self.tau)
		self.target_model.set_weights(tar_w)

	def save_model(self, filename):
		self.model.save(filename)



def main1():
	env = FLORIS(env_settings)
	gamma = 1.0

	episodes= 100
	episode_len = 60
	epsilon = 1.0

	dqn_agent = DQN(env=env)
	steps = []
	rewards= []
	

	for ep in range(episodes):
		episode_rew = 0
		cur_state = env._reset().reshape(1,-1)
		for step in range(episode_len):
			# print('step: %s' %(step)) 
			action = dqn_agent.action(cur_state, epsilon)

			new_state, reward, done, _ = env._step(action)
			new_state = new_state.reshape(1, -1)
			dqn_agent.store(cur_state, action, reward, new_state, done)

			dqn_agent.replay()
			dqn_agent.target_train()

			cur_state = new_state.reshape(1,-1)
			if done:
				break
			
			episode_rew += reward 

		
		rewards.append(episode_rew)
		epsilon = epsilon - (2/episodes) if epsilon > 0.01 else 0.01
		print("Episode:{}\n Reward:{}\n Epsilon:{}".format(ep, episode_rew, epsilon))
	# episode_rew = np.concatenate([i for i in episode_rew], axis=-1)
	
	fig, ax = plt.subplots()
	ax.plot(rewards)
	ax.grid()
	plt.show()

main1()