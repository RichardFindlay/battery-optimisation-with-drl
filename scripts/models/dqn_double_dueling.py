import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque 
from noise_linear import NoisyLinear
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
	def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128, fc4_units=128):
		super(QNet, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.fc_val = nn.Sequential(
			nn.Linear(state_size, fc1_units),
			nn.ReLU(),
			nn.Linear(fc1_units, fc2_units),
			nn.ReLU(),
			# nn.Linear(fc2_units, fc3_units),
			# nn.ReLU(),
			# nn.Linear(fc3_units, fc4_units),
			# nn.ReLU(),
			nn.Linear(fc4_units, 1))

		self.fc_adv = nn.Sequential(
			nn.Linear(state_size, fc1_units),
			nn.ReLU(),
			nn.Linear(fc1_units, fc2_units),
			nn.ReLU(),
			# nn.Linear(fc2_units, fc3_units),
			# nn.ReLU(),
			# nn.Linear(fc3_units, fc4_units),
			# nn.ReLU(),
			nn.Linear(fc4_units, action_size))

	def forward(self, x):
		val = self.fc_val(x)
		adv = self.fc_adv(x)
		return val + adv - adv.mean(dim=1, keepdim=True)


# class NNQNet(nn.Module):
# 	def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128, fc4_units=128):
# 		super(NNQNet, self).__init__()
# 		self.seed = torch.manual_seed(seed)

# 		self.fc_val = nn.Sequential(
# 			nn.Linear(state_size, fc1_units),
# 			nn.ReLU(),
# 			nn.Linear(fc1_units, fc2_units),
# 			nn.ReLU(),
# 			# nn.Linear(fc2_units, fc3_units),
# 			# nn.ReLU(),
# 			# nn.Linear(fc3_units, fc4_units),
# 			# nn.ReLU(),
# 			NoisyLinear(fc4_units, 1))

# 		self.fc_adv = nn.Sequential(
# 			nn.Linear(state_size, fc1_units),
# 			nn.ReLU(),
# 			nn.Linear(fc1_units, fc2_units),
# 			nn.ReLU(),
# 			# nn.Linear(fc2_units, fc3_units),
# 			# nn.ReLU(),
# 			# nn.Linear(fc3_units, fc4_units),
# 			# nn.ReLU(),
# 			NoisyLinear(fc4_units, action_size))

# 	def forward(self, x):
# 		val = self.fc_val(x)
# 		adv = self.fc_adv(x)
# 		return val + adv - adv.mean(dim=1, keepdim=True)


class NNQNet(nn.Module):
	"""Policy Model """
	def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
		super(NNQNet, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.dense1 = NoisyLinear(state_size, fc1_units)
		self.dense2 = NoisyLinear(fc1_units, fc2_units)
		self.dense3 = NoisyLinear(fc2_units, 1)

		self.dense5 = NoisyLinear(state_size, fc1_units)
		self.dense6 = NoisyLinear(fc1_units, fc2_units)
		self.dense7 = NoisyLinear(fc2_units, action_size)

	def forward(self, states):
		"""map state values to action values """
		x1 = F.relu(self.dense1(states))
		x1 = F.relu(self.dense2(x1))
		x1 = self.dense3(x1)

		x2 = F.relu(self.dense5(states))
		x2 = F.relu(self.dense6(x2))
		x2 = self.dense7(x2)

		return x1 + x2 - x2.mean(dim=1, keepdim=True)

	def reset_params(self):

		self.dense2.reset_noise()
		self.dense3.reset_noise()
		self.dense6.reset_noise()
		self.dense7.reset_noise()


# replay buffer object
class Replay():
	def __init__(self, action_size, buffer_size, batch_size, seed):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = self.experiences(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		""" randomly sample experiences from memory """
		experiences = random.sample(self.memory, k=self.batch_size)
      
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().cuda()
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().cuda()
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().cuda()
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().cuda()
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().cuda()

		return (states,actions,rewards,next_states,dones)

	def __len__(self):
		""" get current size of samples in memory """
		return len(self.memory)

# DQN Agent
class DQN_Agent_double_duel():
	def __init__(self, state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update, qnet_type='double_dueling'):
		self.state_size = state_size
		self.action_size = action_size
		self.qnet_type = qnet_type

		# Intialise q-networks
		if self.qnet_type == 'double_dueling':
			self.qnet = QNet(state_size, action_size, seed).cuda()
			self.qnet_target = QNet(state_size, action_size, seed).cuda()
		if self.qnet_type == 'double_dueling_NN':
			self.qnet = NNQNet(state_size, action_size, seed).cuda()
			self.qnet_target = NNQNet(state_size, action_size, seed).cuda()

		# define optimiser
		self.optimizer = optim.Adam(self.qnet.parameters(), lr=learning_rate)

		# Replay Memory 
		self.memory = Replay(action_size, buffer_size, batch_size, seed)
		self.t_step = 0
		self.soft_update_bool = soft_update        

	def step(self, state, action, reward, next_step, update, batch_size, gamma, tau, done):

		# save expereince in model
		self.memory.add(state, action, reward, next_step, done)

		# learn every 'x' time-steps
		self.t_step = (self.t_step+1) % update

		if self.soft_update_bool == True:
			if self.t_step == 0:
				if len(self.memory) > batch_size:
					experience = self.memory.sample()
					self.learn(experience, gamma, tau)
		else:
			if len(self.memory) > batch_size:
				experience = self.memory.sample()
				self.learn(experience, gamma, tau)

	def action(self, state, epsilion = 0):
		""" return action for given state given current policy """
		state = torch.from_numpy(state).float().unsqueeze(0).cuda()
		self.qnet.eval()

		with torch.no_grad():
			action_values = self.qnet(state)

		self.qnet.train()    

		# action selection relative to greedy action selection
		if random.random() > epsilion:
			return np.argmax(action_values.cpu().data.numpy(), axis=1)
		else:
			return random.choice(np.arange(self.action_size))


	def learn(self, experiences, gamma, tau):

		states, actions, rewards, next_states, dones = experiences

		criterion = torch.nn.MSELoss()

		# local model used to train
		self.qnet.train()

		# target model used in eval mode
		self.qnet_target.eval()

		qsa = self.qnet(states).gather(1,actions)

		qsa_prime_actions = self.qnet(next_states).detach().max(1)[1].unsqueeze(1)
        
		qsa_prime_targets = self.qnet_target(next_states).gather(1,qsa_prime_actions)
 
		labels = rewards + (gamma * qsa_prime_targets*(1-dones))

		loss = criterion(qsa, labels).cuda()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# now update the target next weights
		if self.soft_update_bool == True:      
			self.soft_update(self.qnet, self.qnet_target, tau)
		elif (self.soft_update_bool == False) and self.t_step == 0:
			self.soft_update(self.qnet, self.qnet_target, tau=1)

	def soft_update(self, local_model, target_model, tau):

		"""Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target """

		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
















