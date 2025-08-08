#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 05:42:10 2025

@author: localadmin

In this script we will explore some of the properties of the exact solver 
as it scales. This includes:
    - Solution Size as number of consumers or DCs increases.
    - Runtime as number of consumers or DC increases.
    - Number of DCs utilised as number of consumers or DC increases
    - Solution for fixed conditions and varying alpha
    
"""
import numpy as np
import matplotlib.pyplot as plt
from flexmedia_delivery_instance import flexmedia_delivery_instance
import math

#####
#how does the solution scale with consumers?
#####

m = 12
n = np.arange(100, 2050, 50)
sol_varying_consumers = []  #how does objective scale?
time_varying_consumers = [] #how does runtime scale?
utilisation_varying_consumers = [] #how does utilisation scale?
n_reps = 50

for entry in n: #for each number of consumers
    current_sol = 0 #for storing info for this entry
    current_time = 0
    n_datacentres_used_current = 0
    for i in range(n_reps): #for each rep generate and solve instance
        opt_instance = flexmedia_delivery_instance(m, entry, 3, edge_capacity = 200,
                                                   core_capacity = 300)
        opt_instance.exact_solve(1.0)
        current_sol += opt_instance.optimisation_model.ObjVal
        current_time += opt_instance.optimisation_model.Runtime
        n_datacentres_used_current += len(opt_instance.open_centres)/m
    
    #store averages over iterations
    sol_varying_consumers.append(current_sol / n_reps)
    time_varying_consumers.append(current_time / n_reps)
    utilisation_varying_consumers.append(n_datacentres_used_current / n_reps)
    print(entry)

#######
#Plotting Consumers
#######

plt.plot(n, sol_varying_consumers)
plt.title('Cost for Increasing Consumers')
plt.xlabel('Consumers')
plt.ylabel('Average Objective Function (Cost)')
plt.show() 

plt.plot(n, time_varying_consumers)
plt.title('Time for Increasing Consumers')
plt.xlabel('Consumers')
plt.ylabel('Average Runtime (seconds)')
plt.show() 

plt.plot(n, utilisation_varying_consumers)
plt.title('Utilisation for Increasing Consumers')
plt.xlabel('Consumers')
plt.ylabel('Average Utlisation')
plt.show() 

########
#Scaling DCs (although consumers also scales to avoid infeasible instances)
########

m = np.arange(5, 20, 1)
sol_varying_dc = []  #how does objective scale?
time_varying_dc = [] #how does runtime scale?
utilisation_varying_dc = [] #how does utilisation scale?
n_reps = 50

for entry in m: #for each number of consumers
    current_sol = 0 #for storing info for this entry
    current_time = 0
    n_datacentres_used_current = 0
    for i in range(n_reps): #for each rep generate and solve instance
        opt_instance = flexmedia_delivery_instance(m, m*180, 3, edge_capacity = 200,
                                                   core_capacity = 300)
        opt_instance.exact_solve(1.0)
        current_sol += opt_instance.optimisation_model.ObjVal
        current_time += opt_instance.optimisation_model.Runtime
        n_datacentres_used_current += len(opt_instance.open_centres)/entry
    
    #store averages over iterations
    sol_varying_dc.append(current_sol / n_reps)
    time_varying_dc.append(current_time / n_reps)
    utilisation_varying_dc.append(n_datacentres_used_current / n_reps)
    print(entry)
    
#######
#Plotting
#######

plt.plot(n, sol_varying_dc)
plt.title('Cost for Increasing Datacentres')
plt.xlabel('Number of Datacentres')
plt.ylabel('Average Objective Function (Cost)')
plt.show() 

plt.plot(n, time_varying_dc)
plt.title('Time for Increasing Datacentres')
plt.xlabel('Number of Datacentres')
plt.ylabel('Average Runtime (seconds)')
plt.show() 

plt.plot(n, utilisation_varying_dc)
plt.title('Utilisation for Increasing Datacentres')
plt.xlabel('Number of Datacentres')
plt.ylabel('Average Utlisation')
plt.show() 

########
#Vaying Alpha
#########

m = 10
n = 1000
sol_varying_alpha = []  #how does objective scale?
time_varying_alpha = [] #how does runtime scale?
utilisation_varying_alpha = [] #how does utilisation scale?
n_reps = 100

alpha_vec = np.arange(1, 201, step = 5)
for entry in alpha_vec: #for each number of consumers
    current_sol = 0 #for storing info for this entry
    current_time = 0
    n_datacentres_used_current = 0
    for i in range(n_reps): #for each rep generate and solve instance
        opt_instance = flexmedia_delivery_instance(m, n, 3, edge_capacity = 200,
                                                   core_capacity = 300)
        opt_instance.exact_solve(entry)
        current_sol += opt_instance.optimisation_model.ObjVal
        current_time += opt_instance.optimisation_model.Runtime
        n_datacentres_used_current += len(opt_instance.open_centres)/entry
    
    #store averages over iterations
    sol_varying_alpha.append(current_sol / n_reps)
    time_varying_alpha.append(current_time / n_reps)
    utilisation_varying_alpha.append(n_datacentres_used_current / n_reps)
    print(entry)

#######
#Plotting Alpha
#######

plt.plot(n, sol_varying_alpha)
plt.title('Cost for Increasing Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Objective Function (Cost)')
plt.show() 

plt.plot(n, time_varying_alpha)
plt.title('Time for Increasing Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Runtime (seconds)')
plt.show() 

plt.plot(n, utilisation_varying_alpha)
plt.title('Utilisation for Increasing Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Utlisation')
plt.show() 
