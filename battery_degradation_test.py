# Battery Degradation test Script
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



class BatteryDegradation():
	def __init__(self, battery_capacity_watts):
		self.battery_capacity = battery_capacity_watts
		self._params = { # (Kim & Qiao, 2011) : https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1210&context=electricalengineeringfacpub
			'a_0': -0.852, 'a_1': 63.867, 'a_2': 3.6297, 'a_3': 0.559, 'a_4': 0.51, 'a_5': 0.508,
			'b_0': 0.1463, 'b_1': 30.27, 'b_2': 0.1037, 'b_3': 0.0584, 'b_4': 0.1747, 'b_5': 0.1288,
			'c_0': 0.1063, 'c_1': 62.94, 'c_2': 0.0437, 'd_0': -200, 'd_1': -138, 'd_2': 300,
			'e_0': 0.0712, 'e_1': 61.4, 'e_2': 0.0288, 'f_0': -3083, 'f_1': 180, 'f_2': 5088,
			'y1_0': 2863.3, 'y2_0': 232.66, 'c': 0.9248, 'k': 0.0008
			}
		self._ref_volts = 3.6
		self._cellnum = int(np.ceil(self.battery_capacity / self._ref_volts))


	def ss_circuit_model(self, soc):
		v_oc = ((self._params['a_0'] * np.exp(-self._params['a_1'] * soc)) + self._params['a_2'] + (self._params['a_3'] * soc) - (self._params['a_4'] * soc**2) + (self._params['a_5'] * soc**3)) * self._cellnum
		r_s = ((self._params['b_0'] * np.exp(-self._params['b_1'] * soc)) + self._params['b_2'] + (self._params['b_3'] * soc) - (self._params['b_4'] * soc**2) + (self._params['b_5'] * soc**3)) * self._cellnum
		r_st = (self._params['c_0'] * np.exp(-self._params['c_1'] * soc) + self._params['c_2']) * self._cellnum
		r_tl = (self._params['e_0'] * np.exp(-self._params['e_1'] * soc) + self._params['e_2']) * self._cellnum
		r_tot = (r_s + r_st + r_tl)
		return v_oc, r_tot


	def circuit_current(self, v_oc, r_tot, p_r):
		icur = (v_oc - np.sqrt((v_oc**2) - (4 * (r_tot * p_r)))) / (2 * r_tot) 
		return icur


	def calc_efficiency(self, v_oc, r_tot, icur, p_r):
		if p_r > 0:
			efficiency = 1 / ((v_oc - (r_tot * icur)) / v_oc)
		elif p_r < 0:
			efficiency =  v_oc / (v_oc - (r_tot * icur))
		else:
			efficiency = 1.0 

		return efficiency





#####################################################################################################################################
# TEST EFFICIENCY
cap_watts = 100000
soc_all = np.linspace(0,1,11)
p_r_all = np.linspace(0,-cap_watts,11)

v_oc_tot = np.empty((len(soc_all),len(p_r_all)))  
r_tot_tot = np.empty((len(soc_all),len(p_r_all)))    
icur_tot = np.empty((len(soc_all),len(p_r_all)))    
efficiency_tot = np.empty((len(soc_all),len(p_r_all)))    


plot_soc = np.empty((len(soc_all),len(p_r_all)))
plot_p_r = np.empty((len(soc_all),len(p_r_all))) 


batt_degrade = BatteryDegradation(cap_watts)

for idx, soc in enumerate(soc_all):
	for idx2, p_r in enumerate(p_r_all):
		# p_r = p_r 
		print(f'soc: {soc}')
		print(f'power: {p_r}')

		plot_soc[idx, idx2] = soc
		plot_p_r[idx, idx2] = p_r

		v_oc_tot[idx, idx2], r_tot_tot[idx, idx2] = batt_degrade.ss_circuit_model(soc)

		icur_tot[idx, idx2] = batt_degrade.circuit_current(v_oc_tot[idx, idx2], r_tot_tot[idx, idx2], p_r)

		efficiency_tot[idx, idx2] = batt_degrade.calc_efficiency(v_oc_tot[idx, idx2], r_tot_tot[idx, idx2], icur_tot[idx, idx2], p_r)


print(efficiency_tot)
print(efficiency_tot[0,10])
# efficiency_tot[0,10] = 2.0
# X, Y = np.meshgrid(soc_all, p_r_all)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(plot_soc, plot_p_r, efficiency_tot, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax.set_xlabel('soc')
ax.set_ylabel('p_r')
ax.set_zlabel('efficiency');
ax.set_zlim([0.95,1.0])
# ax.set_zlim([1.0,1.06])
plt.show()

exit()