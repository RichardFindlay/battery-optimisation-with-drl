# test / sandbox script to test Pytorch model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import scipy
import sys

from pickle import load
import matplotlib.pyplot as plt

# import model architecture from model scripts directory
sys.path.append('../models') 
from da_electricity_price_model import LSTMCNNModel

# load model weights
PATH = '../models/da_price_prediction.pt'

# intialise model
model = LSTMCNNModel()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# declare train or test set
data_set = 'test'

# load test data
test_load = open(f"../data/processed_data/{data_set}_data_336hr_in_24hr_out_unshuffled.pkl", "rb") 
test_data = load(test_load)
test_load.close()

# ensure model is in evaluation mode
model.eval()

# format inputs into pytorch tensors with appropiate shape
inputs = np.moveaxis(test_data[f'X_{data_set}'], -1, 1)
inputs = torch.tensor(inputs, dtype=torch.float64)

# run prediction model
with torch.no_grad(): 
	prediction = model(inputs.float())

# grab the true values for testing
y_true = test_data[f'y_{data_set}']

# remove additional dimension
y_true = np.squeeze(y_true)
prediction = np.squeeze(prediction)

# load scaler 
scaler = load(open(f'../data/processed_data/da_price_scaler.pkl', 'rb'))

# retain vars sizes for reshaping of results
test_len = len(y_true)
output_seq_size = 24

# inverse transform data
prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(test_len, output_seq_size)
y_true = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(test_len, output_seq_size)

# calculate eval metrics
mae = mean_absolute_error(y_true, prediction)
mape = mean_absolute_percentage_error(y_true, prediction)
rmse = mean_squared_error(y_true, prediction, squared=False)

# declare figure
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
plt.style.use(['seaborn-whitegrid'])

# colour array for plot
colours = ['#f8cf01','#f8a87d', '#96b2c6', '#6e6e6e']

# summer prediction example
idx = 559
axes[0].plot(np.squeeze(prediction[idx:idx+7]).flatten(), label="Prediction", color=colours[1], linewidth=1.2)
axes[0].plot(np.squeeze(y_true[idx:idx+7,:]).flatten(), label="True",  color=colours[2], linewidth=1.2)

# get assicated times
times_axis_1 = test_data[f'y_{data_set}_times'][idx][0]

# summer prediction example
idx = 599
axes[1].plot(np.squeeze(prediction[idx:idx+7]).flatten(), label="Prediction", color=colours[1], linewidth=1.2)
axes[1].plot(np.squeeze(y_true[idx:idx+7,:]).flatten(), label="True",  color=colours[2], linewidth=1.2)

# get assicated times
times_axis_2 = test_data[f'y_{data_set}_times'][idx][0]

# helper function to format axes
def axis_format(axis):
	axis.grid(True, alpha=0.6, which="both")
	axis.spines['bottom'].set_color('black')  
	axis.spines['top'].set_color('black')
	axis.spines['left'].set_color('black')
	axis.spines['right'].set_color('black')
	axis.tick_params(direction="out", length=2.0)
 
	axis.tick_params(axis='y', labelsize= 8)
	axis.tick_params(axis='x', labelsize= 8)
	axis.set_ylabel('Price (£/MWh)', fontsize=9, style='italic', weight='bold')
	axis.set_xlabel('Hour', fontsize=9, style='italic', weight='bold')
	axis.grid(alpha=0.3)

	axis.set_xlim([0, 168])
	# axis.set_ylim([0, 1])
	axis.tick_params(direction="out", length=2.0)
	axis.set_xticks(np.arange(0, 169, 24))

axis_format(axes[0])
axis_format(axes[1])

handle1, label1 = axes[0].get_legend_handles_labels()

leg = axes[1].legend(handle1, label1, loc="lower right", fontsize=7, frameon=True)
leg.set_zorder(5)

frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
# plt.legend(dqn_types, fontsize=8)


# apply additional formatting
axes[0].set_xticklabels([])
axes[0].set_xlabel("")

axes[0].set_title(f"w/c {times_axis_1}", fontsize=9, weight="bold")
axes[1].set_title(f"w/c {times_axis_2}", fontsize=9, weight="bold")

plt.savefig('pytorch_model_performance.png', bbox_inches='tight', transparent=True, dpi=300)
plt.show()

print(mae)
print(mape)
print(rmse)

# find and plot best results
def correlation_analysis(X, Y):

	rs = np.empty((X.shape[0], 1))
	#caclulate 'R^2' for each feature - average over all days
	for l in range(X.shape[0]):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X[l,:], Y[l,:])
		rs[l, 0] =r_value**2
		

	print('mean' + '\n R**2: %s' %rs.mean())
	print('max' + '\n R**2: %s' %rs.max())
	print('min' + '\n R**2: %s' %rs.min())

	# get best and worst result and print
	best_fit = np.argmax(rs, axis=0)
	worst_fit = np.argmin(rs, axis=0)
	print(best_fit)
	print(worst_fit)

	return 

correlation_analysis(y_true, prediction)









