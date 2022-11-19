# script to convert results to d3 graph format
import numpy as np
import pandas as pd

from pickle import load

# choose DQN model type
dqn_model = 'double_dueling' # ['vanilla', 'double_dueling', 'NN', 'double_dueling_NN']

# load results data
timeseries_result_file = open('../../results/dqn_timeseries_results.pkl', 'rb')
timeseries_result_dict = load(timeseries_result_file)

# load time references seperately
time_refs = pd.read_csv('../../data/N2EX_UK_DA_Auction_Hourly_Prices_2018_train.csv', parse_dates=True)['Datetime_UTC']
# time_refs = pd.to_datetime(time_refs)

data = timeseries_result_dict[dqn_model]

# id_ref | date \ price | SoC | action
# id	date	test_data	soc	action

# ensure columns are in correct format 
data['price'] = data['price'].astype(np.float32)
data['action'] = data['action'].astype(np.float32)

data.insert(loc=1, column='date', value=time_refs)

data.to_clipboard()



