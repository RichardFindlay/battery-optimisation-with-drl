# test / sandbox script to test Pytorch model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pickle import load
import matplotlib.pyplot as plt

from DA_ElectricityPrice import LSTMCNNModel

# load model
PATH = './Models/da_price_prediction_336hr_in_24hr_out_model.pt'

model = LSTMCNNModel()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


# load test data
test_load = open("./Data/processed_data/test_data_336hr_in_24hr_out.pkl", "rb") 
test_data = load(test_load)
test_load.close()


model.eval()
loss = 0 

inputs = np.moveaxis(test_data['X_test'], -1, 1)

inputs = torch.tensor(inputs, dtype=torch.float64)
print(test_data['X_test'].shape)
# exit()



with torch.no_grad(): 
	prediction = model(inputs.float())


y_true = test_data['y_test']


idx = 7400
plt.plot(np.squeeze(prediction[idx]), label="pred")
plt.plot(np.squeeze(y_true[idx,:,0]), label="true")
plt.legend()
plt.show()





# with torch.no_grad():
# 	for X in test_data['X_test']:
# 		# a = np.expand_dims(X,axis=0)
# 		x = torch.tensor(np.expand_dims(X,axis=0), dtype=torch.float64)
# 		prediction = model(x.float())
# 		loss += loss_fn(prediction, y.float())
# 	print(f'loss: {loss}')