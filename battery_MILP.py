import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt

from battery_degrade import BatteryDegradation


class BatteryMILP():

	def __init__(self, optimise_df):
		self.df = optimise_df

		# decalre some params used in calculation of effiency


	def optimise(self, power, capacity):

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
		_cellnum = int(np.ceil(capacity/ self._ref_volts))

		# price dictionary
		prices = dict(enumerate(self.df.price.tolist()))
		hourly_refs = dict(enumerate(self.df.hour.tolist()))

		# define problem as pyomo's 'concrete' model
		model = ConcreteModel()

		# set params
		# model.T = Set(doc="hour in simulation", ordered=True, initialize=hourly_refs)
		model.T = RangeSet(0, len(self.df.hour.tolist())-1)
		model.pwr = Param(initialize=power, doc="power rating of battery (MW)")
		model.cap = Param(initialize=capacity, doc="capacity rating of battery (MWh)")
		model.price = Param(model.T, initialize=prices, doc="hourly price (£/MWh)")

		# set charge and discharge varaibles
		model.energy_in = Var(model.T, domain=NonNegativeReals)
		model.energy_out = Var(model.T, domain=NonNegativeReals)


		# set state-of-charge bounds
		model.soc = Var(model.T, bounds=(0, model.cap), initialize=0)

		# set boolean charge and discharge vars
		model.charge_bool= Var(model.T, within=Boolean)
		model.discharge_bool = Var(model.T, within=Boolean, initialize=0)

		# store profit on timeseries resolution
		model.profit_timeseries = Var(model.T, within=Reals, initialize=0)

		# declare Voc and Rtot
		model.voc = Var(model.T, within=Reals, initialize=0)
		model.rtot = Var(model.T, within=Reals, initialize=0)



		# calculate efficenecy - first calualte Voc & R_tot
		def ss_circuit_model(model, t):
			v_oc == ((_params['a_0'] * np.exp(-_params['a_1'] * model.soc[t])) + _params['a_2'] + (_params['a_3'] * model.soc[t]) - (_params['a_4'] * model.soc[t]**2) + (_params['a_5'] * model.soc[t]**3)) * _cellnum
			r_s = ((_params['b_0'] * np.exp(-_params['b_1'] * model.soc[t])) + _params['b_2'] + (_params['b_3'] * model.soc[t]) - (_params['b_4'] * model.soc[t]**2) + (_params['b_5'] * model.soc[t]**3)) * _cellnum
			r_st = (_params['c_0'] * np.exp(-_params['c_1'] * model.soc[t]) + _params['c_2']) * _cellnum
			r_tl = (_params['e_0'] * np.exp(-_params['e_1'] * model.soc[t]) + _params['e_2']) * _cellnum
			r_tot == (r_s + r_st + r_tl)
			return v_oc, r_tot	

		model.voc_rtot = Constraint(model.T, rule=ss_circuit_model)


		efficiency = 0.85



		# state of charge constraint
		def update_soc(model, t):
			if t == model.T.first():
				return model.soc[t] == 0.5
			else:
				return model.soc[t] == model.soc[t-1] - ((efficiency * model.energy_out[t])) + ((efficiency * model.energy_in[t]))

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
			current_profit = (model.energy_out[t] * model.price[t])  - (model.energy_in[t] * model.price[t]) 
			return model.profit_timeseries[t] == current_profit  


		model.profit_track = Constraint(model.T, rule=timeseries_profit)







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


		# define profit
		export_revenue = [self.df.iloc[t-1, 2] * model.energy_out[t] for t in model.T]
		import_cost = [self.df.iloc[t-1, 2] * model.energy_in[t] for t in model.T]
		profit_ts = np.array(export_revenue) - np.array(import_cost)

		profit_obj = np.sum(profit_ts)

		# declare objective
		model.objective = Objective(expr=profit_obj, sense=maximize)


		# declare molde solver and solve
		sol = SolverFactory('glpk')
		sol.solve(model)

		return model



price_data = pd.read_csv('./Data/N2EX_UK_DA_Auction_Hourly_Prices_ts.csv')



# Instaniate MILP battery object with price data
a = BatteryMILP(price_data)

# call optmise method - storing pyomo model 
mod = a.optimise(100, 200)


model_results = {} 

# loop through the vars 
for idx, v in enumerate(mod.component_objects(Var, active=True)):
    print(idx, v.getname())

    var_val = getattr(mod, str(v))

    model_results[f'{v.getname()}'] = var_val[:].value



df = pd.DataFrame(model_results)

df.to_clipboard()


# print(profit.shape)




# get the value for each of variables
# plt.plot(np.array(profit))
# plt.show()



# hours = range(mod.T[0 + 1], mod.T[8000 + 1] + 1)


# print(price_data)

# print(mod.energy_in)

# e_in = [value(mod.energy_in[i]) for i in hours]

# print(e_in)
























