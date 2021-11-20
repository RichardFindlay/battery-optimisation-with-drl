# battery storage optmisation with Reinforcement Learning
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os
from pickle import dump, load

# import custom classes:
from DA_ElectricityPrice import LSTMCNNModel
from battery_degrade import BatteryDegradation


# T is the time period per episode, t is HH periods within T 


# create ESS environment
class Battery(gym.Env):

	def __init__(self, env_settings):

		self.pr = env_settings['battery_energy']
		self.cr = env_settings['battery_capacity']
		self.cost = env_settings['battery_price']
		self.num_actions = env_settings['num_actions']
		self.ep_len = env_settings['num_episodes']
		self.standby_loss = env_settings['standby_loss']
		self.torch_file = env_settings['torch_model']
		self.train = env_settings['train']
		self.train_data_path = env_settings['train_data_path']
		self.test_data_path = env_settings['test_data_path']
		self.scaler_transform_path = env_settings['scaler_transform_path']
		self.alpha_d = 0    							# degradation coefficient
		self.soc = 0.5								# intial state of charge
		self.charging_eff = None						# charging efficiency
		self.dis_charging_eff = None					# dis-charging efficiency
		self.dod = None									# depth of dis-charge
		self.input_prices = None						# to hold train or test samples for training/testing price inputs
		self.ts = 0 									# timestep within each episode
		self.ts_len = 168								# max timestep length
		self.ep = 0   									# episode increment
		self.ep_pwr = 0     							# total absolute power per each episode
		self.kwh_cost = 0    							# battery cost per kwh
		self.ep_start_kWh = 0     						# episode start charge
		self.ep_end_kWh = 0     						# episode end charge
		self.kWh_cost = 104.4     						# battery cost per kWh
		self.price_ref = 0 	 							# counter to keep track of da price index
		self.input_seq_size = 168
		self.ep_prices = np.zeros((24))

		# define parameter limits
		limits = np.array([
			[-self.pr, self.pr],	# battery energy
			[0.0, self.cr],			# battery capacity
			[0.0, 1.0],				# state of charge
			[0.0, 1.0],				# charging efficiency
			[0.0, 1.0],				# discharging efficiency
			[-np.inf, np.inf]		# reward function
		])

		# self.observation_space = gym.spaces.Box(
		# 	low = limits[:,0],
		# 	high = limits[:,1])

		# load scaler for inverse transform
		scaler_load = open(self.scaler_transform_path, "rb") 
		self.scaler_transform = load(scaler_load)
		scaler_load.close()


		self.observation_space = np.append(np.zeros(24), self.soc)

		self.action_space = np.linspace(-1, 1, num = self.num_actions , endpoint = True)

		# intialise degradation class
		self.batt_deg = BatteryDegradation(self.cr * 1000)

		# load DA Price Predictor Pytorch Model
		if os.path.isfile(self.torch_file):
			self.model = LSTMCNNModel()
			self.model.load_state_dict(torch.load(self.torch_file, map_location=torch.device('cpu')))
			self.model.eval()
		else:
			print('Pytorch model not found')
			exit()

		# load test or train price data - should hold in memory (no need to batch)
		if self.train == True:
			train_load = open(self.train_data_path, "rb") 
			self.input_prices = load(train_load)
			train_load.close()
		elif self.train == False:
			test_load = open(self.test_data_path, "rb") 
			self.input_prices = load(test_load)
			test_load.close()


	def _next_soc(self, soc_t, efficiency, action, standby_loss):
		e_ess = self.cr / 360
		if self.pr < 0:
			next_soc = soc_t - (1/e_ess) * efficiency * (action)  
		elif self.pr > 0:
			next_soc = soc_t - (1/e_ess) * (1/efficiency) * (action)  
		else:
			next_soc = soc_t - (1/e_ess) * (standby_loss) 
		return next_soc


	def _get_da_prices(self, input_seq):
		# (samples, channels, input_seq_len)

		input_seq = np.expand_dims(input_seq, axis=0)
		input_seq = np.moveaxis(input_seq, -1, 1)
		input_seq = torch.tensor(input_seq, dtype=torch.float64)

		with torch.no_grad(): 
			predictions = self.model(input_seq.float())

		predictions = torch.squeeze(predictions)
		
		return predictions


	def _degrade_coeff(self):
		print('*********************************************')
		print('degrade')
		print(self.ep_start_kWh)
		print(self.ep_end_kWh)
		print(self.ep_pwr)
		print(self.kWh_cost)
		print('*********************************************')

		self.alpha_d = ((self.ep_start_kWh - self.ep_end_kWh) / self.ep_pwr) * self.kWh_cost
		

	def step(self, state, action):

		# collect current vars from state space
		da_prices = state[:24]
		current_soc = state[-1]
		done = False

		# store episode start capacity 
		if self.ts == 0:
			print(f'current_soc: {current_soc}')
			self.ep_start_kWh = current_soc * self.cr
			# get prices for next episode length
			self.ep_prices = []
			for idx in range((self.ts_len//24) + 1):
				self.ep_prices.append(self._get_da_prices(self.input_prices['X_train'][self.ep + self.price_ref]).numpy())
				self.price_ref += 1
			# combine arrays to create price timeseries episode length
			self.ep_prices = np.concatenate(self.ep_prices)
			# inverse transform predictions
			self.ep_prices = self.scaler_transform.inverse_transform(np.expand_dims(self.ep_prices,axis=-1))
			
		# convert action to kW 
		action_kw = (self.action_space[action] * self.pr)

		# store action to kWh
		action_kwh = action_kw 

		# calculate ohmic, charge and membrance resitances - open circuit voltage & total resitance
		v_oc, r_tot = self.batt_deg.ss_circuit_model(current_soc)

		# calculate circuit current
		icur = self.batt_deg.circuit_current(v_oc, r_tot, action_kw)

		# calculate efficiency of battery relevant to current charge and power 
		efficiency = self.batt_deg.calc_efficiency(v_oc, r_tot, icur, action_kw)

		# state of charge at end of period
		next_soc = self._next_soc(current_soc, efficiency, action_kwh, self.standby_loss)

		# get t+1 price, return (sample, 24hr, 1)
		# da_prices = self._get_da_prices(self.input_prices['X_train'][self.ep + self.ts])

		# reward function for current timestep
		ts_reward =  (self.ep_prices[self.ts] * (action_kw / self.pr)) - (self.alpha_d * (abs(action_kw) / self.pr))

		# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

		# print(self.ep_prices[self.ts])
		# print(action_kw)
		# print(self.pr)
		# print(self.alpha_d)

		# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

		# collect power charge & discharge for episode
		self.ep_pwr += abs(action_kw)

		# update observations
		observations = np.append(self.ep_prices[self.ts:self.ts+24],  next_soc)

		print(f'action: {action}')
		print(f'action_kw: {action_kw}')
		print(observations)
		exit()

		if self.ts == self.ts_len:
			self.ep_end_kWh = next_soc * self.cr
			done = True

		self.soc = next_soc

		# increment timestep
		self.ts += 1 

		return observations, ts_reward, done



	def reset(self):

		# update degrade co-efficient (only if more than one episode) 
		if self.ep > 0:
			self._degrade_coeff()

		# increment epidsodes
		self.ep += 1
		self.ep_pwr = 0

		# reset vars
		self.ts = 0
		self.prev_action = None


		observations = np.append(self.ep_prices[-24:], self.soc)


		return observations













