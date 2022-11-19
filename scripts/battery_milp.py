import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import math

from battery_efficiency import BatteryEfficiency
from battery_degradation_func import calculate_degradation
import itertools


class BatteryMILP():

	def __init__(self, battery_power, battery_capacity):
		# self.df = optimise_df

		self.cap = battery_capacity
		self.pwr = battery_power

		self.previous_cap = 100
		self.batt_cost = 75000 # £/MWh

	def optimise(self, price_df, intial_soc, current_cycle_num, remaining_capacity, previous_ep_power):

		"""
		optimise_df : pandas dataframe containing the hourly price with date range as index
		"""
		
		# price dictionary
		prices = dict(enumerate(price_df.price.tolist()))

		# define problem as pyomo's 'concrete' model
		model = ConcreteModel()

		# set params
		# model.T = Set(doc="hour in simulation", ordered=True, initialize=hourly_refs)
		model.T = RangeSet(0, len(price_df.price.tolist())-1)
		model.pwr = Param(initialize=self.pwr, doc="power rating of battery (MW)")
		model.cap = Param(initialize=self.cap, doc="capacity rating of battery (MWh)")
		model.price = Param(model.T, initialize=prices, doc="hourly price (£/MWh)")

		# set charge and discharge varaibles
		model.energy_in = Var(model.T, domain=NonNegativeReals, initialize=0)
		model.energy_out = Var(model.T, domain=NonNegativeReals, initialize=0)

		# set state-of-charge bounds
		model.soc = Var(model.T, bounds=(0.2*model.cap*(remaining_capacity/100), model.cap*(remaining_capacity/100)), initialize=0.2*model.cap*(remaining_capacity/100))

		# set boolean charge and discharge vars
		model.charge_bool= Var(model.T, within=Boolean, initialize=0)
		model.discharge_bool = Var(model.T, within=Boolean, initialize=0)

		# store profit on timeseries resolution
		model.profit_timeseries = Var(model.T, within=Reals, initialize=0)
		model.cumulative_profit = Var(model.T, within=Reals, initialize=0)

		# declare var for cycle rate
		model.cumlative_cycle_rate = Var(model.T, within=Reals, initialize=0)

		charging_efficiency = 0.923
		discharge_efficiency = 0.923

		# state of charge constraint
		def update_soc(model, t):
			if t == 0:
				return model.soc[t] == intial_soc - ((model.energy_out[t])) + (((model.energy_in[t])))
			else:
				return model.soc[t] == model.soc[t-1] - ((model.energy_out[t])) + (((model.energy_in[t])))

		model.state_of_charge = Constraint(model.T, rule=update_soc)

		# current charge power constraint
		def charge_constraint(model, t):
			return model.energy_in[t] <= (model.pwr)

		model.charge = Constraint(model.T, rule=charge_constraint)

		# current charge power constraint
		def discharge_constraint(model, t):
			return model.energy_out[t] <= (model.pwr)

		model.discharge = Constraint(model.T, rule=discharge_constraint)

		def timeseries_profit(model, t):
			current_profit = (model.energy_out[t] * model.price[t] * discharge_efficiency)  - (model.energy_in[t] * model.price[t] / charging_efficiency) 
			return model.profit_timeseries[t] == current_profit  

		model.profit_track = Constraint(model.T, rule=timeseries_profit)

		# use constraint to calculate cumlative cycle rate at each timestep
		def cycle_rate_per_ts(model,t):
			ts_cycle = ((model.energy_out[t] + model.energy_in[t]) / self.pwr) / 2
			if t == 0:
				return model.cumlative_cycle_rate[t] == ts_cycle + current_cycle_num
			else:
				return model.cumlative_cycle_rate[t] == model.cumlative_cycle_rate[t-1] + ts_cycle

		model.cycles = Constraint(model.T, rule=cycle_rate_per_ts)

		# define constraint for cumlative profit
		def cumlative_profit(model, t):
			if t == 0:
				return model.cumulative_profit[t] == (model.energy_out[t] * model.price[t] * discharge_efficiency) - (model.energy_in[t] * model.price[t] / charging_efficiency)
			else:
				return model.cumulative_profit[t] == model.cumulative_profit[t-1] + ((model.energy_out[t] * model.price[t] * discharge_efficiency) - (model.energy_in[t] * model.price[t] / charging_efficiency))

		model.all_profit = Constraint(model.T,rule=cumlative_profit)

		# calulcate degradation costs
		if previous_ep_power != 0:
			alpha_degradation = ((((self.previous_cap - remaining_capacity)/100) * self.cap) / previous_ep_power) * self.batt_cost
		else:
			alpha_degradation = 0 

		# get degradation cost
		degrade_cost = [alpha_degradation * (model.energy_out[t] + model.energy_in[t]) for t in model.T]

		# define profit
		export_revenue = [price_df.iloc[t, -1] * model.energy_out[t] * discharge_efficiency for t in model.T]
		import_cost = [price_df.iloc[t, -1] * model.energy_in[t] / charging_efficiency for t in model.T]
		profit_ts = np.array(export_revenue) - np.array(import_cost) - np.array(degrade_cost)

		profit_obj = np.sum(profit_ts)

		# declare objective function
		model.objective = Objective(expr=profit_obj, sense=maximize)

		# implement bigM constraint to ensure model doesn't simultaneously charge and discharge
		def Bool_char_rule_1(model, t):
		    bigM=5000000
		    return((model.energy_in[t])>=-bigM*(model.charge_bool[t]))
		model.Batt_ch1=Constraint(model.T,rule=Bool_char_rule_1)

		# if battery is charging, charging must be greater than -large
		# if not, charging geq zero
		def Bool_char_rule_2(model, t):
		    bigM=5000000
		    return((model.energy_in[t])<=0+bigM*(1-model.discharge_bool[t]))
		model.Batt_ch2=Constraint(model.T,rule=Bool_char_rule_2)

		# if batt discharging, charging must be leq zero
		# if not, charging leq +large
		def Bool_char_rule_3(model, t):
		    bigM=5000000
		    return((model.energy_out[t])<=bigM*(model.discharge_bool[t]))
		model.Batt_cd3=Constraint(model.T,rule=Bool_char_rule_3)

		# if batt discharge, discharge leq POSITIVE large
		# if not, discharge leq 0
		def Bool_char_rule_4(model, t):
		    bigM=5000000
		    return((model.energy_out[t])>=0-bigM*(1-model.charge_bool[t]))
		model.Batt_cd4=Constraint(model.T,rule=Bool_char_rule_4)

		# if batt charge, discharge geq zero
		# if not, discharge geq -large
		def Batt_char_dis(model, t):
		    return (model.charge_bool[t]+model.discharge_bool[t],1)
		model.Batt_char_dis=Constraint(model.T,rule=Batt_char_dis)

		# declare molde solver and solve
		sol = SolverFactory('glpk')
		sol.solve(model)

		return model

	def update_remaining_cap(self, cycle_num):
		remaining_cap = calculate_degradation(cycle_num)
		return remaining_cap


price_data = pd.read_csv('../data/N2EX_UK_DA_Auction_Hourly_Prices_2019_test.csv')

# declare battery config
battery_power = 10
battery_capacity = 20

# declare intial soc
soc = 0.5 * battery_capacity
current_cycle = 0
remaining_capacity = 100
previous_ep_power = 0

# Instaniate MILP battery object with price data
a = BatteryMILP(battery_power, battery_capacity)

# pass daily prices for optmisation
for day_idx in range(0, len(price_data), 168):
	print(day_idx)

	# call optmise method - storing pyomo model 
	mod = a.optimise(price_data[day_idx:day_idx+168], soc, current_cycle, remaining_capacity, previous_ep_power)

	model_results = {} 

	# loop through the vars 
	for idx, v in enumerate(mod.component_objects(Var, active=True)):
		# print(idx, v.getname())

		var_val = getattr(mod, str(v))

		model_results[f'{v.getname()}'] = var_val[:].value

	# store data in dataframe
	if day_idx == 0:
		df = pd.DataFrame(model_results)
	else:
		df = pd.concat([df, pd.DataFrame(model_results)])

	# store soc for next 'episode'
	soc = df['soc'].iloc[-1]

	# store cycle to carry forward for cumlative calculation
	current_cycle = df['cumlative_cycle_rate'].iloc[-1]

	a.previous_cap = remaining_capacity

	# update alpha degradation after 'episode'
	remaining_capacity = a.update_remaining_cap(current_cycle)

	previous_ep_power = abs(np.sum(df['energy_in'].iloc[-168:] + df['energy_out'].iloc[-168:]))


# update cumlative profit so ensure profits carried between episodes
df['cumlative_profit'] = df['profit_timeseries'].cumsum()

# save profits for runtime duration (for comparison with DQN models)
df.to_csv('../results/timeseries_results_MILP.csv')

plt.plot(df['cumlative_profit'].values)
plt.show()

df.to_clipboard()























