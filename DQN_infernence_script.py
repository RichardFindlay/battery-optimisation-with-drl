import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt 

from DA_ElectricityPrice import LSTMCNNModel
from battery_degrade import BatteryDegradation
from battery_environment import Battery
from DQN_pytorch import DQN_Agent

# declare environment dictionary
env_settings = {
	'battery_capacity': 5000,	# rated capacity of battery (kWh)
    'battery_energy': 1000,		# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 5,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.99,		# standby loss for battery when idle
    'num_episodes': 1000,		# number of episodes 
    'train': True,				# Boolean to determine whether train or test state
    'scaler_transform_path': './Data/processed_data/da_price_scaler.pkl',				
    'train_data_path': './Data/processed_data/train_data_336hr_in_24hr_out_unshuffled.pkl', # Path to trian data
    'test_data_path': './Data/processed_data/test_data_336hr_in_24hr_out_unshuffled.pkl',	 # Path to test data
    'torch_model': './Models/da_price_prediction_336hr_in_24hr_out_model.pt'	 # relevant to current file dir
}


test_size = 100
time_range = 168

state_size = (25)
action_size = 5

learning_rate = 25e-5 
buffer_size = int(1e5)
batch_size = 32 # 64 best
gamma = 0.99
# tau = 1e-3
tau = 1
update = 10000 # 168 best also 100 for hard up date 
seed = 100

dqn_agent = DQN_Agent(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed)
dqn_agent.qnet.load_state_dict(torch.load('checkpoint.pth', map_location=torch.device('cpu')))




env = Battery(env_settings)


total_cumlative_profit = 0
cumlative_profit = [0]

for test_eps in range(test_size):
	print(f'day_ref: {test_eps}')

	cur_state = env.reset()
	
	for step in range(time_range): 

		action = dqn_agent.action(cur_state)

		new_state, reward, done, info = env.step(cur_state, action, step)

		total_cumlative_profit += info["ts_cost"]
		cumlative_profit.append(cumlative_profit[-1] + info["ts_cost"])


print(total_cumlative_profit)

fig, ax = plt.subplots()
ax.plot(cumlative_profit, label="cumlative_profit")
ax.legend()
ax.grid()
plt.show()

