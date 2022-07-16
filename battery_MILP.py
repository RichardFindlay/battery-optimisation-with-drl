import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import math

from battery_degrade import BatteryDegradation
from battery_degradation_func import calculate_degradation
import itertools

class BatteryMILP():

	def __init__(self, battery_power, battery_capacity):
		# self.df = optimise_df

		self.cap = battery_capacity
		self.pwr = battery_power

		self.previous_cap = 100
		self.batt_cost = 75000

		# self.intial_soc = 0.2 * self.cap


	def optimise(self, price_df, intial_soc, current_cycle_num, remaining_capacity, previous_ep_power):

		"""
		optimise_df : pandas dataframe containing the hourly price with date range as index
		"""
		_params = { # (Kim & Qiao, 2011) : https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1210&context=electricalengineeringfacpub
			'a_0': -0.852, 'a_1': 63.867, 'a_2': 3.6297, 'a_3': 0.559, 'a_4': 0.51, 'a_5': 0.508,
			'b_0': 0.1463, 'b_1': 30.27, 'b_2': 0.1037, 'b_3': 0.0584, 'b_4': 0.1747, 'b_5': 0.1288,
			'c_0': 0.1063, 'c_1': 62.94, 'c_2': 0.0437, 'd_0': -200, 'd_1': -138, 'd_2': 300,
			'e_0': 0.0712, 'e_1': 61.4, 'e_2': 0.0288, 'f_0': -3083, 'f_1': 180, 'f_2': 5088,
			'y1_0': 2863.3, 'y2_0': 232.66, 'c': 0.9248, 'k': 0.0008
			}
		_ref_volts = 3.6

		# calculate number of cells for capactiy 
		_cellnum = int(np.ceil(self.cap / _ref_volts))

		# price dictionary
		prices = dict(enumerate(price_df.price.tolist()))
		# prices = dict(itertools.islice(prices.items(), 21))
		# hourly_refs = dict(enumerate(self.df.hour.tolist()))

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

		# decalre var for alpha
		# model.alpha_degradation = Var(model.T, within=Reals, initialize=0)

		# declare Voc and Rtot
		# model.voc = Param(model.T, within=Reals, initialize=0)
		# model.rtot = Var(model.T, within=Reals, initialize=0)

		# model.rs = Var(model.T, within=Reals, initialize=0)
		# model.st = Var(model.T, within=Reals, initialize=0)
		# model.tl = Var(model.T, within=Reals, initialize=0)

		charging_efficiency = 0.923
		discharge_efficiency = 0.923

		# def boolean_constraint(model, t):
		# 	return (model.charge_bool[t] + model.discharge_bool[t]) <= 1

		# model.charnge_discharge_bool = Constraint(model.T, rule=boolean_constraint)

		# boolen constraint 1
		# def boolean_constraint1(model, t):
		# 	return (model.charge_bool[t] + model.discharge_bool[t])	<= 1

		# model.boolean_constraint = Constraint(model.T, rule=boolean_constraint1)


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


		# use constraint to calculate alpha 
		def alpha_degrade(model,t):
			if previous_ep_power != 0:
				return model.alpha_degradation[t] == ((((self.previous_cap - remaining_capacity)/100) * self.cap) / previous_ep_power) * self.batt_cost
			else:
				return model.alpha_degradation[t] == 0

		# model.degrade = Constraint(model.T, rule=alpha_degrade)

		# calculate efficiency voc at each timestep





		# # calculate efficiency - first calualte Voc & R_tot
		# def ss_circuit_model(model, t):
		# 	voc = ((_params['a_0'] * exp(-_params['a_1'] * model.soc[t])) + _params['a_2'] + (_params['a_3'] * model.soc[t]) - (_params['a_4'] * model.soc[t]**2) + (_params['a_5'] * model.soc[t]**3)) * _cellnum
		# 	return model.voc[t] == voc

		# # model.voc_rtot = Constraint(model.T, rule=ss_circuit_model)

		# def rtot_calc(model, t):
		# 	return model.rs[t] == ((_params['b_0'] * math.exp(-_params['b_1'] * model.soc[t].value)) + _params['b_2'] + (_params['b_3'] * model.soc[t].value) - (_params['b_4'] * model.soc[t].value**2) + (_params['b_5'] * model.soc[t].value**3)) * _cellnum
		# 	# model.st[t] == (_params['c_0'] * np.exp(-_params['c_1'] * model.soc[t].value) + _params['c_2']) * _cellnum
		# 	# model.tl[t] == (_params['e_0'] * np.exp(-_params['e_1'] * model.soc[t].value) + _params['e_2']) * _cellnum

		# 	# return model.rtot[t] == (model.rs[t] + model.st[t] + model.tl[t])

		# # model.rtot_constraint = Constraint(model.T, rule=rtot_calc)


		# # calculate open circuit voltage and total resistence at time 't'
		# def ss_circuit_model(self, model, t):
		# 	v_oc = ((self._params['a_0'] * np.exp(-self._params['a_1'] * model.soc[t])) + self._params['a_2'] + (self._params['a_3'] * model.soc[t]) - (self._params['a_4'] * model.soc[t]**2) + (self._params['a_5'] * model[t].soc**3)) * self._cellnum
		# 	r_s = ((self._params['b_0'] * np.exp(-self._params['b_1'] * model.soc[t])) + self._params['b_2'] + (self._params['b_3'] * model.soc[t]) - (self._params['b_4'] * model.soc[t]**2) + (self._params['b_5'] * model[t].soc**3)) * self._cellnum
		# 	r_st = (self._params['c_0'] * np.exp(-self._params['c_1'] * model.soc[t]) + self._params['c_2']) * self._cellnum
		# 	r_tl = (self._params['e_0'] * np.exp(-self._params['e_1'] * model.soc[t]) + self._params['e_2']) * self._cellnum
		# 	r_tot = (r_s + r_st + r_tl)
		# 	return v_oc, r_tot	

		# model.v_oc, model.r_tot = Constraint(model.T, rule=ss_circuit_model)

		# # method to calculate current
		# def circuit_current(self, model, t):
		# 	if model.charge[t] > 0:
		# 		icur = (model.v_oc[t] - np.sqrt((model.v_oc[t]**2) - (4 * (model.r_tot[t] * model.charge[t])))) / (2 * model.r_tot[t]) 
		# 	elif model.discharge[t] > 0:
		# 		icur = (model.v_oc[t] - np.sqrt((model.v_oc[t]**2) - (4 * (model.r_tot[t] * model.discharge[t])))) / (2 * model.r_tot[t]) 
		# 	return icur

		# model.icur = Constraint(model.T, rule=circuit_current)

		# # define method for calculate efficency
		# def calc_efficiency(self, model, t):
		# 	if model.charge[t] > 0:
		# 		efficiency = 1 / ((model.v_oc[t] - (model.r_tot[t] * model.icur[t])) / model.v_oc[t])
		# 	elif model.discharge[t] > 0:
		# 		efficiency =  model.v_oc[t] / (model.v_oc[t] - (model.r_tot[t] * model.icur[t]))
		# 	else:
		# 		efficiency = 1.0

		# 	return efficiency 		

		# model.effiency = Constraint(model.T, rule=calc_efficiency)


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



price_data = pd.read_csv('./Data/N2EX_UK_DA_Auction_Hourly_Prices_2015_train.csv')

battery_power = 10
battery_capacity = 20


# price_data = price_data[:336]

# declare intial soc
soc = 0.5 * battery_capacity
current_cycle = 0
remaining_capacity = 100
previous_ep_power = 0

# Instaniate MILP battery object with price data
a = BatteryMILP(battery_power, battery_capacity)

print(len(price_data))

# pass daily prices for optmisation
for day_idx in range(0, len(price_data), 168):
	print(day_idx)

	# Instaniate MILP battery object with price data
	# a = BatteryMILP(price_data[day_idx:day_idx+168])

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



df.to_clipboard()























