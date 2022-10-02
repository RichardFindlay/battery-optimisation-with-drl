import pandas as pd
from pickle import load 
import matplotlib.pyplot as plt
import seaborn as sns


# set graph style
plt.style.use(['seaborn-whitegrid'])


# comparison plot between cumlative profits over test year

# load results from dqn inference testing
cumsum_dqn = open('./results/dqn_cumlativeprofit_results.pkl', 'rb')
cumsum_dqn = load(cumsum_dqn)

# load results from MILP
cumsum_milp = pd.read_csv('./results/timeseries_results_MILP.csv')
cumsum_milp = cumsum_milp['cumlative_profit']

# legend list
legend_ls = ['Vanilla', 'Double Dueling', 'Noisy Network', 'MILP']

# colours list
colours = ['dodgerblue', 'tomato', 'green', 'black']

# plot cumlative profits for milp and DQN together 
for idx, dqn_mod in enumerate(cumsum_dqn.keys()):
	plt.plot(cumsum_dqn[dqn_mod], color=colours[idx])


# intialise plot axis
ax = plt.subplot()


# apply graph formatting
ax.grid(True, alpha=0.6, which="both")
ax.spines['bottom'].set_color('black')  
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(direction="out", length=2.0)

ax.tick_params(axis='y', labelsize= 8)
ax.tick_params(axis='x', labelsize= 8)
ax.set_ylabel('Profit (£)', fontsize=9, style='italic', weight='bold')
ax.set_xlabel('Hour', fontsize=9, style='italic', weight='bold')
ax.grid(alpha=0.3)


# add milp to plot
plt.plot(cumsum_milp, color = colours[-1])
plt.legend(legend_ls)
plt.show()







