import pandas as pd
from pickle import load 
import matplotlib.pyplot as plt

# comparison plot between cumlative profits over test year

# load results from dqn inference testing
cumsum_dqn = open('./results/dqn_cumlativeprofit_results.pkl', 'rb')
cumsum_dqn = load(cumsum_dqn)

# load results from MILP
cumsum_milp = pd.read_csv('./results/timeseries_results_MILP.csv')
cumsum_milp = cumsum_milp['cumlative_profit']

# plot cumlative profits for milp and DQN together 
for dqn_mod in cumsum_dqn.keys():
	plt.plot(cumsum_dqn[dqn_mod])


plt.plot(cumsum_milp)
plt.show()