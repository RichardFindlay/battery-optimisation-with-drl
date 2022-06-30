import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump

from workalendar.europe import UnitedKingdom
cal = UnitedKingdom()

# create train and test set data
n2ex_da = pd.read_csv('./Data/N2EX_UK_DA_Auction_Hourly_Prices_2015_v3.csv', header=0)


# convert series to datetime
n2ex_da['utc_timestamp'] = pd.to_datetime(n2ex_da['utc_timestamp'],format='%d/%m/%Y %H:%M')

# set datetime as index
# n2ex_da.set_index(n2ex_da['utc_timestamp'], inplace=True)
# n2ex_da.index = pd.to_datetime(n2ex_da['utc_timestamp'], format = '%d/%m/%Y %H:%M')


# remove outliers IQR Method
# q15 = n2ex_da.quantile(0.15).values
# q85 = n2ex_da.quantile(0.85).values
# iqr = q85 - q15
# cut_off = 1.5 * iqr
# lower = float(q15 - cut_off)
# upper = float(q85 + cut_off)

# applying clipping
# n2ex_da['Price_(£)'].clip(lower=lower, upper=upper, inplace=True) 

# normalise the price
scaler = MinMaxScaler()
n2ex_da[['Price_(£)']] = scaler.fit_transform(n2ex_da[['Price_(£)']])

# save scaler for inverse transform
with open(f"./Data/processed_data/da_price_scaler.pkl", "wb") as scaler_store:
	dump(scaler, scaler_store)

# # split data into days i
# days_df =[]
# for group in n2ex_da.groupby(n2ex_da.index.date):
#     days_df.append(group[1])

# # remove days with any nan values
# idx = 0
# for d in range(len(days_df)):
# 	if days_df[idx].isnull().values.any():
# 		del days_df[idx] 
# 		idx -= 1
# 	idx += 1 

# ts_df = pd.concat(days_df)
ts = n2ex_da['Price_(£)']
ts = np.expand_dims(ts.values, axis=-1)

dates = n2ex_da.index
dates = np.array(dates, dtype = 'datetime64[ns]')

print(n2ex_da['utc_timestamp'].values)
# dates.to_clipboard()

# data engineering
df_times = pd.DataFrame()
df_times['hour'] = n2ex_da['utc_timestamp'].dt.hour
df_times['day'] = n2ex_da['utc_timestamp'].dt.day
df_times['month'] = n2ex_da['utc_timestamp'].dt.month 
df_times['day_of_year'] = n2ex_da['utc_timestamp'].dt.dayofyear 
df_times['day_of_week'] = n2ex_da['utc_timestamp'].dt.dayofweek 
# df_times['year'] = n2ex_da.index.year 
df_times['weekend'] = df_times['day_of_week'].apply(lambda x: 1 if x>=5 else 0)


start_year = int(n2ex_da['utc_timestamp'].dt.year.min())
end_year = int(n2ex_da['utc_timestamp'].dt.year.max())

start_date = n2ex_da['utc_timestamp'].min()
end_date = n2ex_da['utc_timestamp'].max()

# add holidays boolean indicator
holidays = set(holiday[0] 
	for year in range(start_year, end_year + 1) 
	for holiday in cal.holidays(year)
	if start_date <=  holiday[0] <= end_date)

df_times['holiday'] = n2ex_da['utc_timestamp'].isin(holidays).astype(int)

# df_times['price'] = ts


# normalise features in each column
for col_idx, col in enumerate(df_times.columns.values[:-2]): # ignore the last two colums as one-hot encoding
	df_times[col] = (df_times[col] - np.min(df_times[col]))  / (np.max(df_times[col]) - np.min(df_times[col]))


# create sin / cos of input times
# times_out_hour_sin = np.expand_dims(np.sin(2*np.pi*df_times['hour']/np.max(df_times['hour'])), axis=-1)
# times_out_month_sin = np.expand_dims(np.sin(2*np.pi*df_times['month']/np.max(df_times['month'])), axis=-1)

# times_out_hour_cos = np.expand_dims(np.cos(2*np.pi*df_times['hour']/np.max(df_times['hour'])), axis=-1)
# times_out_month_cos = np.expand_dims(np.cos(2*np.pi*df_times['month']/np.max(df_times['month'])), axis=-1)

# times_out_year = np.expand_dims((df_times['year'].values - np.min(df_times['year'])) / (np.max(df_times['year']) - np.min(df_times['year'])), axis=-1)

# times_data = np.concatenate((times_out_hour_sin, times_out_hour_cos, times_out_month_sin, times_out_month_cos, times_out_year), axis=-1)

times_data = df_times.values

# print(times_data.shape)

# exit()

# group days for input and output train/test data set
def input_output(ts, times_data, dates, input_seq_size, output_seq_size):
	x_input, x_times_data, y_output, in_times, out_times  = [], [], [], [], []
	input_start = 0
	output_start = input_seq_size

	while (output_start + output_seq_size) < len(ts):

		x_time = np.empty(((input_seq_size)), dtype = 'datetime64[ns]')
		y_time = np.empty(((output_seq_size)), dtype = 'datetime64[ns]')

		input_end = input_start + input_seq_size
		output_end = output_start + output_seq_size

		input_seq = ts[input_start:input_end]
		x_input.append(input_seq)
		eng_times_data = times_data[input_start:input_end]
		x_times_data.append(eng_times_data)
		output_seq = ts[output_start:output_end]
		y_output.append(output_seq)
	
		x_time[:] = dates[input_start:input_end]
		in_times.append(x_time)
		y_time[:] = dates[output_start:output_end]
		out_times.append(y_time)

		input_start += 1 
		output_start += 1

		# steps for test data set
		# input_start += output_seq_size 
		# output_start += output_seq_size

	x_input = np.array(x_input)
	x_times_data = np.array(x_times_data)
	y_output = np.array(y_output)
	x_input_times = np.array(in_times, dtype = 'datetime64[ns]')
	y_output_times = np.array(out_times, dtype = 'datetime64[ns]')

	x_input = np.concatenate([x_input, x_times_data], axis=-1)

	X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.1998859229, shuffle=True) # split used to capture 2019 only

	X_train_times, X_test_times, y_train_times, y_test_times = train_test_split(x_input_times, y_output_times, test_size=0.1998859229, shuffle=True)

	train_data = {
		'X_train': X_train,
		'y_train': y_train,
		'X_train_times': X_train_times,
		'y_train_times': y_train_times
	}

	test_data = {
		'X_test': X_test,
		'y_test': y_test,
		'X_test_times': X_test_times,
		'y_test_times': y_test_times
	}

	print(*[f'{key}: {train_data[key].shape}' for key in train_data.keys()], sep='\n')
	print(*[f'{key}: {test_data[key].shape}' for key in test_data.keys()], sep='\n')

	return train_data, test_data




train_data, test_data = input_output(ts, times_data, dates, input_seq_size=168, output_seq_size=24)


# save data
with open(f"./Data/processed_data/train_data_336hr_in_24hr_out_shuffled.pkl", "wb") as trainset:
	dump(train_data, trainset)

with open("./Data/processed_data/test_data_336hr_in_24hr_out_shuffled.pkl", "wb") as testset:
	dump(test_data, testset)




