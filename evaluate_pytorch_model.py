# test / sandbox script to test Pytorch model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import scipy

from pickle import load
import matplotlib.pyplot as plt


from DA_ElectricityPrice import LSTMCNNModel




# load model
PATH = './Models/da_price_prediction.pt'

model = LSTMCNNModel()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


data_set = 'test'

# load test data
test_load = open(f"./Data/processed_data/{data_set}_data_336hr_in_24hr_out_unshuffled_test.pkl", "rb") 
test_data = load(test_load)
test_load.close()

print(test_data.keys())

print(test_data['X_test'].shape)

# print(test_data['y_train_times'])

model.eval()
loss = 20 

print(test_data.keys())

inputs = np.moveaxis(test_data[f'X_{data_set}'], -1, 1)

inputs = torch.tensor(inputs, dtype=torch.float64)

print(inputs.shape)


with torch.no_grad(): 
	prediction = model(inputs.float())



y_true = test_data[f'y_{data_set}']




y_true = np.squeeze(y_true)
prediction = np.squeeze(prediction)

# load scaler 
scaler = load(open(f'./Data/processed_data/da_price_scaler.pkl', 'rb'))


test_len = len(y_true)
output_seq_size = 24

prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(test_len, output_seq_size)
y_true = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(test_len, output_seq_size)


mae = mean_absolute_error(y_true, prediction)
mape = mean_absolute_percentage_error(y_true, prediction)
rmse = mean_squared_error(y_true, prediction, squared=False)


# declare figure
plt.figure(figsize=(12, 3))
# set graph style
plt.style.use(['seaborn-whitegrid'])

# intialise plot axis
ax = plt.subplot()

colours = ['#f8cf01','#f8a87d', '#96b2c6', '#6e6e6e']

idx = 20
plt.plot(np.squeeze(prediction[idx:idx+7]).flatten(), label="Prediction", color=colours[1], linewidth=1.5)
plt.plot(np.squeeze(y_true[idx:idx+7,:]).flatten(), label="True",  color=colours[2], linewidth=1.5)
# plt.legend()
# plt.show()

# apply graph formatting
ax.grid(True, alpha=0.6, which="both")
ax.spines['bottom'].set_color('black')  
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(direction="out", length=2.0)

ax.tick_params(axis='y', labelsize= 8)
ax.tick_params(axis='x', labelsize= 8)
ax.set_ylabel('Price (£/MWh)', fontsize=9, style='italic', weight='bold')
ax.set_xlabel('Hour', fontsize=9, style='italic', weight='bold')
ax.grid(alpha=0.3)

ax.set_xlim([0, 168])
ax.tick_params(direction="out", length=2.0)
ax.set_xticks(np.arange(0, 169, 24))

# ax.set_ylim([-2, 4])
ax.tick_params(direction="out", length=2.0)
# ax.set_yticks(np.arange(-2, 4.1, 1))

handle1, label1 = ax.get_legend_handles_labels()

leg = ax.legend(handle1, label1, loc="upper right", fontsize=7, frameon=True)
leg.set_zorder(5)

frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
# plt.legend(dqn_types, fontsize=8)

plt.savefig('pytorch_model_performance.png', bbox_inches='tight', transparent=True, dpi=300)
plt.show()



exit()




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

	#get best
	best_fit = np.argmax(rs, axis=0)
	worst_fit = np.argmin(rs, axis=0)
	print(best_fit)
	print(worst_fit)
	# print(X[best_fit,:,0])

	return 



print(y_true.shape)
print(prediction.shape)


correlation_analysis(y_true, prediction)









