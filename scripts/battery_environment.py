# battery storage optmisation with Reinforcement Learning
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os
from pickle import dump, load
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import custom classes:
from battery_efficiency import BatteryEfficiency
from battery_degradation_func import calculate_degradation

# import model architecture from model scripts directory
sys.path.append('../models') 
from da_electricity_price_model import LSTMCNNModel

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
		self.ep_end_kWh_remain = 0     					# episode end charge
		self.kWh_cost = 50     							# battery cost per kWh
		self.price_ref = 0 	 							# counter to keep track of da price index
		self.input_seq_size = 168
		self.done = False
		self.total_ts = 0
		self.day_num = 0
		self.game_over = False
		self.idx_ref = 1
		self.bug = False
		self.true_prices = pd.read_csv('../data/N2EX_UK_DA_Auction_Hourly_Prices_2018_train.csv').iloc[:,-1]
		self.price_track = env_settings['price_track']
		self.cycle_num = 0
		self.ep_start_kWh_remain = self.cr
		self.forecast_index = 0 # index to keep track of forecasting horizon inputs
		self.forecast_bool = False
        
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

		# obs = np.append(self.ts, self.soc)
		self.observation_space = np.append(np.zeros(24), self.soc)

		self.action_space = np.linspace(-1, 1, num = self.num_actions , endpoint = True)

		# intialise degradation class
		self.batt_deg = BatteryEfficiency(self.cr * 1000) # watts as input

		# load DA Price Predictor Pytorch Model
		if os.path.isfile(self.torch_file):
			self.model = LSTMCNNModel().cuda()
			self.model.load_state_dict(torch.load(self.torch_file, map_location=torch.device('cuda:0')))
			self.model = self.model.cuda()
			self.model.eval()
		else:
			print('Pytorch model not found')
			sys.exit()

		# load test or train price data - should hold in memory (no need to batch)
		if self.train == True:
			train_load = open(self.train_data_path, "rb") 
			self.input_prices = load(train_load)
			train_load.close()
		elif self.train == False:
			test_load = open(self.test_data_path, "rb") 
			self.input_prices = load(test_load)
			test_load.close()

		da_inputs = torch.tensor(self.input_prices['X_train'][0], dtype=torch.float64)
		self.ep_prices = self._get_da_prices(da_inputs.float())
		self.ep_prices = self.ep_prices.cpu().data.numpy()

		self.true_scaler = MinMaxScaler()
		self.true_prices = np.squeeze(self.true_scaler.fit_transform(np.expand_dims(self.true_prices,axis=-1)))

		# save true price training scaler
		with open(f"/content/drive/My Drive/Battery-RL/train_true_scaler.pkl", "wb") as scaler_store:
			dump(self.true_scaler, scaler_store)

		# self.ep_prices.append(self._get_da_prices(self.input_prices['X_train'][1]).numpy())
		# self.ep_prices = np.concatenate(self.ep_prices)

	def _next_soc(self, soc_t, efficiency, action, standby_loss):
		e_ess = self.cr
		if np.around(soc_t, 2) == 0 and action == 0:
			next_soc = soc_t
		elif action < 0:
			next_soc = soc_t - (1/e_ess) * efficiency * (action)  
		elif action > 0:
			next_soc = soc_t - (1/e_ess) * (1/efficiency) * (action)  
		elif action == 0:
			next_soc = soc_t - (1/e_ess) * (standby_loss) 
		return next_soc


	def _get_da_prices(self, input_seq):
		# (samples, channels, input_seq_len)

		input_seq = np.expand_dims(input_seq, axis=0)
		input_seq = np.moveaxis(input_seq, -1, 1)
		input_seq = torch.tensor(input_seq, dtype=torch.float64)
		input_seq = input_seq.cuda()

		with torch.no_grad(): 
			predictions = self.model(input_seq.float())

		predictions = torch.squeeze(predictions)
		
		return predictions


	def _degrade_coeff(self):

		if self.ep_pwr == 0:
			self.alpha_d = 0
		else:
			self.alpha_d = ((self.ep_start_kWh_remain - self.ep_end_kWh_remain) / self.ep_pwr) * self.kWh_cost
		
	def step(self, state, action, step):

		# collect current vars from state space
		da_prices = state[:24]
		current_soc = state[-1]
		
		# store episode start capacity 
		if self.new_ep == True:
			start_ep_capacity = calculate_degradation(self.cycle_num)
			self.ep_start_kWh_remain = (start_ep_capacity/100) * self.cr

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

		# clip charge / discahrge relatiev to SoC limits
		upper_lim = ((current_soc - 0.2) * self.cr) / (efficiency * 1)
		lower_lim = ((current_soc - 1) * self.cr) / (efficiency * 1)

		# clip action to ensure within limits
		action_kWh_clipped = np.clip(action_kwh, lower_lim,  upper_lim)

		# round action to 4 sig figures
		# action_kWh_clipped = np.around(action_kWh_clipped, 3)

		# state of charge at end of period
		next_soc = self._next_soc(current_soc, efficiency, action_kWh_clipped, self.standby_loss)

		# detemine if charge/discharge is out of bounds, i.e. <20% and >100% else fail episode
		if np.around(next_soc,2) < 0.2 or np.around(next_soc,2) > 1.0:
			print('error - limits breached')

		# get t+1 price, return (sample, 24hr, 1)
		# da_prices = self._get_da_prices(self.input_prices['X_train'][self.ep + self.ts])

		# reward function for current timestep
		if self.price_track == 'forecasted': 
			ts_price_MW = self.scaler_transform.inverse_transform(np.expand_dims(self.ep_prices[self.ts:self.ts+1],axis=-1))
		if self.price_track == 'true': 
			ts_price_MW = self.true_scaler.inverse_transform(np.expand_dims(self.ep_prices[self.ts:self.ts+1],axis=-1))

		ts_price_kW = ts_price_MW / 1000
		action_MW_clipped = action_kWh_clipped /1000
		ts_reward = np.squeeze((ts_price_kW * (action_kWh_clipped)) - (self.alpha_d * (abs(action_kWh_clipped))))

		# collect power charge & discharge for episode
		self.ep_pwr += abs(action_kWh_clipped)

		# keep track of cycle number for degradation
		self.cycle_num += (abs(action_kWh_clipped) / self.cr) / 2

		# update observations
		price_index_start = self.ts
		price_index_end = self.ts + 24

		observations = np.append(self.ep_prices[price_index_start:price_index_end],  np.around(next_soc,4))

		if step == self.ts_len - 1:
			print('_______________________________')
			# ts_reward +=  50
			# self.ep_end_kWh = next_soc * self.cr
			# self.ts -= 1
			self.done = True

		self.soc = next_soc
		self.new_ep = False

		# increment timestep
		self.ts += 1
		self.total_ts += 1

		# scale reward for quicker learning
		ts_reward = np.around(ts_reward/1000,4)

		ts_cost = np.squeeze((ts_price_kW * (action_kWh_clipped)) - (self.alpha_d * abs(action_kWh_clipped)))

		# store data in dictionary
		info = {'ts_cost': ts_cost}

		return observations, ts_reward, self.done, info




	def reset(self):
		# update final charge for episode
		end_ep_capacity = calculate_degradation(self.cycle_num)
		self.ep_end_kWh_remain = (end_ep_capacity/100) * self.cr

		# update degrade co-efficient (only if more than one episode) 
		# if self.ep > 0:
		self._degrade_coeff()

		# reintialise episode power to zero 
		self.ep_pwr = 0

		# assume battery refreshed after 4000 cycles
		if self.cycle_num >= 4000:      
			self.cycle_num = 0
			# self._degrade_coeff()

		if self.game_over == True:
			self.soc = 0.5

		# ####### OLD METHOD ######################

		# if self.price_track == 'forecasted':
		# 	self.ep_prices = []
		# 	for idx in range((self.ts_len // 24) + 1):
        #     	# deal with reset during training
		# 		if (self.idx_ref + idx) == len(self.input_prices['X_train']):
		# 			self.idx_ref = -idx
		# 			self.bug = True
		# 			# self.cycle_num = 0

		# 		da_inputs = torch.tensor(self.input_prices['X_train'][self.idx_ref + idx], dtype=torch.float64)
		# 		self.ep_prices.append(self._get_da_prices(da_inputs).cpu().data.numpy())
		# 	# combine arrays to create price timeseries episode length
		# 	self.ep_prices = np.concatenate(self.ep_prices)
		# 	# update index ref for next price grab
		# 	self.idx_ref = self.idx_ref + idx

		# elif self.price_track == 'true':

        #     # deal with reset during training
		# 	if (self.idx_ref)+192 >= len(self.true_prices):
		# 		self.idx_ref = 1
		# 		self.cycle_num = 0
                
		# 	self.ep_prices = self.true_prices[(self.idx_ref-1):(self.idx_ref-1)+192]

		# 	# update index ref for next price grab
		# 	self.idx_ref = self.idx_ref + 169


		####### NEW METHOD ######################
		self.forecast_bool = False

        # organise training data for next episode
		if (self.idx_ref+192) >= len(self.true_prices):
			self.idx_ref = 1
			# self.forecast_index = 0
			self.cycle_num = 0

		self.ep_prices = self.true_prices[(self.idx_ref-1):(self.idx_ref-1)+192]

		# get forecasted prices if 'forecasting' true, use real prices until forecast horizon available
		if self.price_track == 'forecasted' and self.idx_ref >= 384: # need to be able to predict the next 8-days
			self.forecast_bool = True
			self.forecast_ep_prices = []
			self.forecast_time_feats = []

			for idx in range((self.ts_len // 24) + 1):

				# fetch time features for current date range
				time_features = torch.tensor(time_engineering(self.input_dates[self.forecast_index: self.forecast_index + 168]), dtype=torch.float64)

				da_inputs = torch.tensor(np.expand_dims(self.true_prices[self.forecast_index: self.forecast_index + 168], axis=-1), dtype=torch.float64)
				# print(f'DA_inputs: {torch.isnan(da_inputs).any()}')

				# concat time features with prev generation for model input
				model_input = torch.cat((da_inputs, time_features), axis=-1)

				self.forecast_ep_prices.append(self._get_da_prices(model_input).cpu().data.numpy())

				self.forecast_index = self.forecast_index + 24

			# combine prices into array
			self.forecast_ep_prices = np.concatenate(self.forecast_ep_prices)

		# update index ref for next price grab
		self.idx_ref = self.idx_ref + 168

		# reset timestep ref
		self.ts = 0

		if self.forecast_bool == True:
			observations = np.append(self.forecast_ep_prices[self.ts:self.ts+24], np.around(self.soc,4))			
		else:
			observations = np.append(self.ep_prices[self.ts:self.ts+24], np.around(self.soc,4))	

		# reintialise episode parameters
		self.done = False
		self.game_over = False
		self.new_ep = True
		self.ep += 1		

		return observations













