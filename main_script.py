# battery storage optmisation with Reinforcement Learning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
		self.alpha_d = 0    							# degradation coefficient
		self.soc_z = 0.5								# intial state of charge
		self.charging_eff = None						# charging efficiency
		self.dis_charging_eff = None					# dis-charging efficiency
		self.dod = None									# depth of dis-charge
		self.input_prices = None						# to hold train or test samples for training/testing price inputs
		self.ts = 0 									# timestep within each episode
		self.ts_len = 168								# max timestep length
		self.ep = 0   									# episode increment
		self.ep_pwr = []     							# total absolute power per each episode
		self.kwh_cost = 0    							# battery cost per kwh

		# define parameter limits
		limits = np.array([
			[-self.p_r, self.p_r],	# battery energy
			[0.0, self.c_r],		# battery capacity
			[0.0, 1.0],				# state of charge
			[0.0, 1.0],				# charging efficiency
			[0.0, 1.0],				# discharging efficiency
			[-np.inf, np.inf]		# reward function
		])

		self.observation_space = spaces.Box(
			low = limits[:,0],
			high = limits[:,1])

		self.action_space = np.linspace(-1, 1, num = self.num_actions , endpoint = True)

		# intialise degradation class
		self.batt_deg = BatteryDegradation(self.cr * 1000)

		# load DA Price Predictor Pytorch Model
		if os.path.isfile(file_name):
			self.model = LSTMCNNModel()
			self.model.load_state_dict(torch.load(self.torch_file, map_location=torch.device('cpu')))
			self.model.eval()
		else:
			print('Pytorch model not found')
			exit()

		# load test or train price data - should hold in memory (no need to batch)
		if self.train == True:
			self.input_prices= open(self.train_data_path, "rb") 
			test_data = load(test_load)
			test_load.close()
		elif self.train == False:
			self.input_prices = open(self.test_data_path, "rb") 
			test_data = load(test_load)
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

		input_seq = np.moveaxis(input_seq, -1, 1)
		input_seq = torch.tensor(input_seq, dtype=torch.float64)

		with torch.no_grad(): 
			predictions = model(input_seq.float())
		
		return predictions


	def _degrade_coeff(self):

		self.alpha_d = ((start_cap - end_cap) / np.sum(self.ep_pwr)) * self.kWh_cost
		

	def step(self, state, action):

		# collect current vars from state space
		da_prices = state[0]
		current_soc = state[-1]

		# convert action to kW 
		action_kw = (action * self.pr)

		# convert action to kWh dvide by 2 for HH
		action_kwh = action_kw / 2

		# calculate ohmic, charge and membrance resitances - open circuit voltage & total resitance
		v_oc, r_tot = self.batt_deg.ss_circuit_model(current_soc)

		# calculate circuit current
		icur = self.batt_deg.circuit_current(v_oc, r_tot, action_kw)

		# calculate efficiency of battery relevant to current charge and power 
		efficiency = self.batt_deg.calc_efficiency(v_oc, r_tot, icur, action_kw)

		# state of charge at end of period
		next_soc = self._next_soc(current_soc, efficiency, action_kwh, self.standby_loss)

		# get t+1 price, return (sample, 24hr, 1)
		da_prices = self._get_da_prices(self.input_prices[self.ep + self.ts])

		# reward function for current timestep
		ts_reward =  (da_prices[self.ts] * (action_kw / self.pr)) * (self.alpha_d * (abs(action_kw) / self.pr))

		# collect power charge & discharge for episode
		self.ep_pwr.append(action_kw)

		# update observations
		obervations = (tuple(da_prices),  next_soc)

		if self.ts == self.ts_len:
			done = True


		return observation, reward, done









	def reset(self):
		# increment epidsodes
		self.ep += 1

		# update degrade co-efficient 
		self._degrade_coeff(self)

		# reset vars
		self.ts = 0
		self.prev_action = None

		self.observation = np.array([])



		return self.observation






# declare environment dictionary
env_settings = {
	'battery_capacity': 10,		# rated capacity of battery (kWh)
    'battery_energy': 10,		# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 6,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.98		# standby loss for battery when idle
    'num_episodes': 1000,		# number of episodes 
    'train': True				# Boolean to determine whether train or test state
    'train_data_path': './Data/processed_data/train_data_336hr_in_24hr_out.pkl', # Path to trian data
    'test_data_path': './Data/processed_data/test_data_336hr_in_24hr_out.pkl',	 # Path to test data
    'torch_model': './Data/processed_data/test_data_336hr_in_24hr_out.pkl'		 # relevant to current file dir

}






