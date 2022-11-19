# CNN-LSTM Pytorch model to predcit DA electricity price
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pickle import load
import matplotlib.pyplot as pl


# check for gpu compatibility
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


# define DA price forecating model in Pytorch
class LSTMCNNModel(nn.Module):
	def __init__(self):
		super(LSTMCNNModel ,self).__init__()
		self.cnn = nn.Conv1d(8, 24, kernel_size=24, stride=24)
		self.lstm = nn.LSTM(7, 32, 6, batch_first=True)
		self.fc = nn.Linear(32, 1)
		self.relu = nn.ReLU()
		# self.device = device

	def forward(self, x):
		out = self.relu(self.cnn(x))
		out, states = self.lstm(out)
		out = self.fc(out)
		return out



def main():

	#load training data dictionary
	train_set_load = open("../data/processed_data/train_data_336hr_in_24hr_out_2018_2019.pkl", "rb") 
	train_set = load(train_set_load)
	train_set_load.close()


	# import data
	inputs = np.moveaxis(train_set['X_train'], -1, 1)
	y_true = train_set['y_train']
	# inputs = np.squeeze(train_set['X_train'],axis=-1)
	# y_true = np.squeeze(train_set['y_train'], axis=-1)

	inputs = torch.tensor(inputs, dtype=torch.float64)
	y_true = torch.tensor(y_true, dtype=torch.float64)

	# inputs, y_true = inputs.cuda(), y_true.cuda()

	# print(train_set['X_train'].shape)
	# print(inputs.shape)
	# print(y_true.shape)

	model = LSTMCNNModel()

	loss_fn = nn.MSELoss(reduction='sum')

	learning_rate = 1e-3
	optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
	epochs = 100

	# train model 
	for idx in range(epochs):
		# make forward pass
		y_pred = model(inputs.float())

		# calc loss 
		loss = loss_fn(y_pred, y_true.float()) / len(y_true)
		if idx % 100 == 0:
			print(f'Epoch: {idx}: loss: {loss.item()}')

		optimiser.zero_grad()

		# backwards pass
		loss.backward()

		# update parameters of optmisier
		optimiser.step()


	# save trained model
	PATH = '../models/da_price_prediction_2018.pt'
	torch.save(model.state_dict(), PATH)



	model.eval()
	with torch.no_grad():
		a = model(torch.unsqueeze(inputs[500].float(),0))
		a = a.numpy()
		a = np.squeeze(a, axis=-1)

	print(a.shape)
	print(y_true[500].shape)
	plt.plot(np.squeeze(a), label="pred")
	plt.plot(np.squeeze(y_true[500]), label="true")
	plt.show()

	# inference model
	# model.eval()
	# loss = 0 

	# with torch.no_grad():
	# 	for X, y in zip(x_test, y_test):
	# 		prediction = model(X)
	# 		loss += loss_fn(prediction, y.float())


if __name__ == "__main__":
   main()




