# battery storage optmisation with Reinforcement Learning
import torch
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
	'battery_capacity': 5000,	# rated capacity of battery (kWh)
    'battery_energy': 1000,		# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 5,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.99,		# standby loss for battery when idle
    'num_episodes': 1000,		# number of episodes 
    'train': True,				# Boolean to determine whether train or test state
    'scaler_transform_path': '/content/drive/My Drive/Battery-RL/Data/processed_data/da_price_scaler.pkl',				
    'train_data_path': '/content/drive/My Drive/Battery-RL/Data/processed_data/train_data_336hr_in_24hr_out_unshuffled.pkl', # Path to trian data
    'test_data_path': '/content/drive/My Drive/Battery-RL/Data/processed_data/test_data_336hr_in_24hr_out_unshuffled.pkl',	 # Path to test data
    'torch_model': '/content/drive/My Drive/Battery-RL/Models/da_price_prediction_336hr_in_24hr_out_model.pt'	 # relevant to current file dir
}

# no clipping of reward signal
# and -500 for exceeding 0-1 boundary SoC

n_episodes = 12000 # max 67925
time_range = 168


epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9996
# epsilon_decay = 0.99965

env = Battery(env_settings)
state_size = (env.observation_space.shape[0])
action_size = len(env.action_space)
seed = 100

learning_rate = 25e-5 
buffer_size = int(1e5)
batch_size = 32 # 64 best
gamma = 0.99
# tau = 1e-3
tau = 1
update = 10000 # 168 best also 100 for hard up date 


dqn_agent = DQN_Agent(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update=False)
scores = np.empty((n_episodes)) #list of rewards from each episode
profits = np.empty((n_episodes))



for ep in range(n_episodes):
	print('NEW EPISODE--------------++++++++++++++++++++++++++++++++++++++_____________')
	episode_rew = 0
	episode_profit = 0

	cur_state = env.reset()
	# print(f'episode: {ep}') 
	
	for step in range(time_range): 
		# print('NEW TS')
		# print(f'step: {step}') 
		# print(f'total_step: {env.total_ts}') 
		# print(f'day_num: {env.day_num}') 
		# print(cur_state)

		action = dqn_agent.action(cur_state, epsilon)
		# print(f'action: {action}') 
		new_state, reward, done, info = env.step(cur_state, action, step)

		# print(new_state)

		dqn_agent.step(cur_state, action, reward, new_state, update, batch_size, gamma, tau, done)

		cur_state = new_state 
		episode_rew += reward

		# store episode profit 
		episode_profit += info["ts_cost"]

		# print(f'reward****@{step}: {episode_rew}')
		# if ep == 3:
		# 	exit()

		if done:
			break

	scores[ep] = episode_rew 
	profits[ep] = episode_profit
	# epsilon = epsilon - (2/(ep+1)) if epsilon > 0.01 else 0.01
	epsilon = max(epsilon*epsilon_decay, epsilon_end)
	# env.ep += 1


	print(f"Episode:{ep}\n Reward:{episode_rew}\n Epsilon:{epsilon}\n Profit: {episode_profit}")

mean_rewards = np.zeros(n_episodes)
for t in range(n_episodes):
	mean_rewards[t] = np.mean(scores[max(0, t-100):(t+1)])


torch.save(dqn_agent.qnet.state_dict(),'checkpoint.pth')

fig, ax = plt.subplots()
ax.plot(scores, label="episode reward")
ax.plot(mean_rewards, label="mean")
ax.legend()
ax.grid()
plt.show()


 



