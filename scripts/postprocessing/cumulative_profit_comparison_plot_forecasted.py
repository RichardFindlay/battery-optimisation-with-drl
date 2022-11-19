import pandas as pd
import numpy as np
from pickle import load 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# comparison plot between cumulative profits over test year using forecasted functionaility of env

# declare figure
plt.figure(figsize=(12, 3))
# set graph style
plt.style.use(['seaborn-whitegrid'])

# load results from dqn inference testing - relative to forecasted prices
cumsum_dqn_forecasted_prices = open('../../results/dqn_cumlativeprofit_results_forecasted.pkl', 'rb')
cumsum_dqn_forecasted_prices = load(cumsum_dqn_forecasted_prices)

# load results from dqn inference testing - relative to true prices
cumsum_dqn_true_prices = open('../../results/dqn_cumlativeprofit_results_forecasted_trueprices.pkl', 'rb')
cumsum_dqn_true_prices = load(cumsum_dqn_true_prices)

# legend list
legend_ls = ['Vanilla', 'Double Dueling', 'Noisy Network']

# colours list
colours = ['#f8cf01','#f8a87d', '#96b2c6', '#6e6e6e']
# colours = ['#6e6e6e', '#457b9d', '#f79256', '#f08080']

# plot cumlative profits for milp and DQN together 
for idx, dqn_mod in enumerate(cumsum_dqn_forecasted_prices.keys()):
	plt.plot(cumsum_dqn_forecasted_prices[dqn_mod], color=colours[idx], linewidth=1.75, label=f"{legend_ls[idx]} forecasted prices")

# plot cumlative profits for milp and DQN together 
for idx, dqn_mod in enumerate(cumsum_dqn_true_prices.keys()):
	plt.plot(cumsum_dqn_true_prices[dqn_mod], color=colours[idx], linewidth=1.75, linestyle='--', dashes=(10, 4), alpha=0.4, label=f"{legend_ls[idx]} true prices")

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
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.grid(alpha=0.3)

ax.set_xlim([0, 8760])
ax.tick_params(direction="out", length=2.0)
ax.set_xticks(np.arange(0, 8760, 1000))

handle1, label1 = ax.get_legend_handles_labels()

leg = ax.legend(handle1, label1, loc="upper left", fontsize=7, frameon=True, handlelength=4)
leg.set_zorder(5)
# ax.legend()

frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

# plt.legend(legend_ls, fontsize=8)
plt.savefig('cumulative_profit_comparison_forecasted.png', bbox_inches='tight', transparent=True, dpi=300)

plt.show()


