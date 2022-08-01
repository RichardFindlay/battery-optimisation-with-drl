import pandas as pd
import numpy as np



# helper fucntion to pass times data and retrieve features egnineering
def time_engineering(input_times_df):
	'''
	input: pandas dataframe with 'utc_timestamp' column in datetime format
	output: numpy array of normalised / engineerd time features
	'''

	# data engineering
	df_times = pd.DataFrame()
	df_times['hour'] = input_times_df['utc_timestamp'].dt.hour
	df_times['day'] = input_times_df['utc_timestamp'].dt.day
	df_times['month'] = input_times_df['utc_timestamp'].dt.month 
	df_times['day_of_year'] = input_times_df['utc_timestamp'].dt.dayofyear 
	df_times['day_of_week'] = input_times_df['utc_timestamp'].dt.dayofweek 
	# df_times['year'] = n2ex_da.index.year 
	df_times['weekend'] = df_times['day_of_week'].apply(lambda x: 1 if x>=5 else 0)

	start_year = int(input_times_df['utc_timestamp'].dt.year.min())
	end_year = int(input_times_df['utc_timestamp'].dt.year.max())

	start_date = input_times_df['utc_timestamp'].min()
	end_date = input_times_df['utc_timestamp'].max()

	# add holidays boolean indicator
	holidays = set(holiday[0] 
		for year in range(start_year, end_year + 1) 
		for holiday in cal.holidays(year)
		if start_date <=  holiday[0] <= end_date)

	df_times['holiday'] = input_times_df['utc_timestamp'].isin(holidays).astype(int)

	# normalise features in each column
	for col_idx, col in enumerate(df_times.columns.values[:-2]): # ignore the last two colums as one-hot encoding
		df_times[col] = (df_times[col] - np.min(df_times[col]))  / (np.max(df_times[col]) - np.min(df_times[col]))

	times_engineered = df_times.values

	return times_engineered
