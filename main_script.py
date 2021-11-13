# battery storage optmisation with Reinforcement Learning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# import custom classes:
from DA_ElectricityPrice import LSTMCNNModel
from battery_degrade import BatteryDegradation
from battery_environment import Battery
from DQN_pytorch import DQN_Agent


# declare environment dictionary
env_settings = {
	'battery_capacity': 10,		# rated capacity of battery (kWh)
    'battery_energy': 10,		# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 5,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.98,		# standby loss for battery when idle
    'num_episodes': 1000,		# number of episodes 
    'train': True,				# Boolean to determine whether train or test state
    'train_data_path': './Data/processed_data/train_data_336hr_in_24hr_out.pkl', # Path to trian data
    'test_data_path': './Data/processed_data/test_data_336hr_in_24hr_out.pkl',	 # Path to test data
    'torch_model': './Models/da_price_prediction_336hr_in_24hr_out_model.pt'		 # relevant to current file dir
}



n_episodes = 200
time_range = 168

gamma = 1.0
epsilon = 1.0


env = Battery(env_settings)
state_size = env.observation_space.shape[0]
action_size = len(env.action_space)
seed = 100

learning_rate = 5e-4 
buffer_size = int(1e5) 
batch_size = 64
gamma = 0.99
tau = 1e-3
update = 4


dqn_agent = DQN_Agent(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed)
scores= [] #list of rewards from each episode




for ep in range(n_episodes):
	episode_rew = 0
	cur_state = env.reset()
	print(f'episode: {ep}') 

	for step in range(time_range):
		print(f'step: {step}') 
		action = dqn_agent.action(cur_state, epsilon)
		new_state, reward, done = env.step(cur_state, action)

		dqn_agent.step(cur_state, action, reward, new_state, update, batch_size, gamma, done)

		cur_state = new_state
		episode_rew += reward 

		if done:
			break

	scores.append(episode_rew)
	# epsilon = epsilon - (2/episodes) if epsilon > 0.01 else 0.01
	epsilon = max(eps*eps_decay,eps_end)

	print("Episode:{}\n Reward:{}\n Epsilon:{}".format(ep, episode_rew, epsilon))

torch.save(dqn_agent.qnet.state_dict(),'checkpoint.pth')

fig, ax = plt.subplots()
ax.plot(scores)
ax.grid()
plt.show()






