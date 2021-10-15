import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pickle import dump




# create train and test set data
n2ex_da = pd.read_csv('./Data/N2EX_UK_DA_Auction_Hourly_Prices.csv', header=0)


# convert series to datetime
n2ex_da['Datetime_CET'] = pd.to_datetime(n2ex_da['Datetime_CET'])

# set datetime as index
n2ex_da.set_index(n2ex_da['Datetime_CET'], inplace=True)

# remove outliers IQR Method
q15 = n2ex_da.quantile(0.15).values
q85 = n2ex_da.quantile(0.85).values
iqr = q85 - q15
cut_off = 1.5 * iqr
lower = float(q15 - cut_off)
upper = float(q85 + cut_off)

# applying clipping
n2ex_da['Price_(£)'].clip(lower=lower, upper=upper, inplace=True) 

# normalise the price
scaler = MinMaxScaler()
n2ex_da[['Price_(£)']] = scaler.fit_transform(n2ex_da[['Price_(£)']])

# split data into days i
days_df =[]
for group in n2ex_da.groupby(n2ex_da.index.date):
    days_df.append(group[1])

# remove days with any nan values
idx = 0
for d in range(len(days_df)):
	if days_df[idx].isnull().values.any():
		del days_df[idx] 
		idx -= 1
	idx += 1 

ts_df = pd.concat(days_df)
ts = ts_df.values

dates = ts_df.index.values
dates = np.array(dates, dtype = 'datetime64[ns]')



# group days for input and output train/test data set
def input_output(ts, dates, input_seq_size, output_seq_size):
	x_input, y_output, in_times, out_times  = [], [], [], []
	input_start = 0
	output_start = input_seq_size

	while (output_start + output_seq_size) < len(ts):

		x = np.empty((input_seq_size, 1))
		y = np.empty((output_seq_size, 1))

		x_time = np.empty(((input_seq_size)), dtype = 'datetime64[ns]')
		y_time = np.empty(((output_seq_size)), dtype = 'datetime64[ns]')


		input_end = input_start + input_seq_size
		output_end = output_start + output_seq_size

		input_seq = ts[input_start:input_end]
		x_input.append(input_seq)
		output_seq = ts[output_start:output_end]
		y_output.append(output_seq)
	

		x_time[:] = dates[input_start:input_end]
		in_times.append(x_time)
		y_time[:] = dates[output_start:output_end]
		out_times.append(y_time)

		input_start += 1 
		output_start += 1

	x_input = np.array(x_input)
	y_output = np.array(y_output)
	x_input_times = np.array(in_times, dtype = 'datetime64[ns]')
	y_output_times = np.array(out_times, dtype = 'datetime64[ns]')

	print(x_input.shape)


	X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.1, random_state=28)

	X_train_times, X_test_times, y_train_times, y_test_times = train_test_split(x_input_times, y_output_times, test_size=0.1, random_state=28)

	train_data = {
		'X_train': X_train,
		'y_train': y_train,
		'X_train_times': X_train_times,
		'y_train_times': y_train_times
	}

	test_data = {
		'X_train': X_test,
		'y_train': y_test,
		'X_test_times': X_test_times,
		'y_test_times': y_test_times
	}


	return train_data, test_data




train_data, test_data = input_output(ts, dates, input_seq_size=168, output_seq_size=24)

# save data
with open(f"./Data/processed_data/train_data_168hr_in_24hr_out.pkl", "wb") as trainset:
	dump(train_data, trainset)

with open("./Data/processed_data/test_data_168hr_in_24hr_out.pkl", "wb") as testset:
	dump(test_data, testset)




