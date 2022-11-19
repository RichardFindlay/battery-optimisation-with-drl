def calculate_degradation(cycle_num):
	''' representative degradiation curve derived from https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment'''
	remaining_cap = 0.000000000000000000093031387249*cycle_num**6 - 0.00000000000000135401195479516*cycle_num**5 + 0.00000000000769793660702074*cycle_num**4 - 0.0000000215754356367543*cycle_num**3 + 0.000031144155854923*cycle_num**2 - 0.025873264406556*cycle_num + 100
	return remaining_cap
