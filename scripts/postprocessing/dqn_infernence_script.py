import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pickle import dump

# import model architecture from model scripts directory
sys.path.append('./models') 
sys.path.append('../models') 

# import custom classes:
from da_electricity_price_model import LSTMCNNModel
from battery_efficiency import BatteryEfficiency
from battery_environment import Battery
from dqn_vanilla import DQN_Agent
from dqn_double_dueling import DQN_Agent_double_duel

# declare environment dictionary
env_settings = {
	'battery_capacity': 20000,	# rated capacity of battery (kWh)
    'battery_energy': 10000,	# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 5,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.99,		# standby loss for battery when idle
    'num_episodes': 1000,		# number of episodes 
    'train': True,				# Boolean to determine whether train or test state
    'scaler_transform_path': './Data/processed_data/da_price_scaler.pkl',				
    'train_data_path': './Data/processed_data/train_data_336hr_in_24hr_out_unshuffled.pkl', # Path to trian data
    'test_data_path': './Data/processed_data/test_data_336hr_in_24hr_out_unshuffled.pkl',	 # Path to test data
    'torch_model': './Models/da_price_prediction.pt',	 # relevant to current file dir
    'price_track': 'forecasted' # forecasted / true
}

test_size = 52 # number of weeks to run test for
time_range = 168

state_size = (25)
action_size = 5

learning_rate = 25e-5 
buffer_size = int(1e5)
batch_size = 32 # 64 best
gamma = 0.99
# tau = 1e-3
tau = 0.001
update = 16 # 168 best also 100 for hard up date 
seed = 100

# list all dqn variations
dqn_types = ['vanilla', 'double_dueling', 'NN']

# store profits for each model in dicitonary
dqn_model_profits = {}

# store timeseries performance in pandas df for each model type
timeseries_performance = {}

# loop over all dqn variations
for model in dqn_types:

	# instaniate DQN agent
	if model == "double_dueling" or model == "double_dueling_NN":
		dqn_agent = DQN_Agent_double_duel(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update=True, qnet_type=model)
	else:
		print(model)
		dqn_agent = DQN_Agent(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update=True, qnet_type=model)        

	dqn_agent.qnet.load_state_dict(torch.load(f'../../models/dqn/dqn_{model}.pth', map_location=torch.device('cpu')))

	env = Battery(env_settings)

	total_cumlative_profit = 0
	cumlative_profit = [0]

	actions = []
	socs = []
	prices = []

	hour = []
	idx = 0

	for test_eps in range(test_size):
		print(f'day_ref: {test_eps}')

		cur_state = env.reset()
		
		for step in range(time_range): 

			action = dqn_agent.action(cur_state)

			hour.append(idx)
			socs.append(cur_state[-1])

			new_state, reward, done, info = env.step(cur_state, action, step)

			actions.append(info['action_clipped'])
			prices.append(info['price'])

			total_cumlative_profit += info["ts_cost"]
			cumlative_profit.append(cumlative_profit[-1] + info["ts_cost"])

			cur_state = new_state

			idx += 1 

	dqn_model_profits[model] = cumlative_profit

	# create dataframe to show timeseries performance
	timeseries_performance[model] = pd.DataFrame({'hour': hour, 'price': prices, 'soc': socs, 'action': actions})


# plot profit comparison
for idx, model in enumerate(dqn_types):
	plt.plot(dqn_model_profits[model], label=f'{model}')

plt.legend(loc="lower right")
plt.show()

print(timeseries_performance.keys())

# save timeseries performance df via pickle
with open('../../results/dqn_timeseries_results_forecasted_trueprices.pkl', 'wb') as timeseries_results:
	dump(timeseries_performance, timeseries_results)

# save cumlative profits for each dqn model
with open('../../results/dqn_cumlativeprofit_results_forecasted_trueprices.pkl', 'wb') as profits:
	dump(dqn_model_profits, profits)


# timeseries_performance['double_dueling'].to_clipboard()




