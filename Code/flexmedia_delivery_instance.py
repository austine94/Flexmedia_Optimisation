#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:23:09 2025

@author: localadmin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import gurobipy as gp
from gurobipy import GRB

from random_datacentres import random_datacentres
from random_consumers import random_consumers
from random_costs import random_datacentre_costs, random_consumer_costs, random_datacentre_capacities, random_consumer_latencies

class flexmedia_delivery_instance:
    
    '''
    This is a class to store information about a single instance of the 
    flexmedia delivery optimisation problem
    
    We have n_datacentres and n_consumers and the consumers are randomly
    drawn from a mixture of Bivariate Gaussians - all this is over [0,1]^2.
    
    The min_radius is the min gap between data centres - this is ensured using
    rejection sampling.
    
    The min and max var are the ranges for the min and max variances of the
    bivariate Gaussian hotspot centres - note the variance is circular,
    there is no correlation parameter.
    '''
    
    def __init__(self, n_datacentres, n_consumers, n_hotspots, 
                 min_radius = 0.01, min_var = 0.01, max_var = 0.1,
                 edge_min = 10, edge_max = 100, core_min = 1, core_max = 10,
                 consumer_min = 0.01, consumer_max = 1, 
                 edge_capacity = 10, core_capacity = 100):
        self.datacentre_locations = random_datacentres(n_datacentres,
                                                       min_radius)
        self.consumer_locations = random_consumers(n_consumers, n_hotspots,
                                                   min_var, max_var)
        self.datacentre_costs = random_datacentre_costs(self.datacentre_locations,
                                                        edge_min, edge_max, 
                                                        core_min, core_max)
        self.consumer_costs = random_consumer_costs(n_consumers, n_datacentres,
                                                    consumer_min, consumer_max)
        self.datacentre_capacities = random_datacentre_capacities(self.datacentre_locations, 
                                                                  edge_capacity,
                                                                  core_capacity)
        self.consumer_latencies = random_consumer_latencies(self.datacentre_locations,
                                                            self.consumer_locations,
                                                            1.0)
        self.optimisation_model = None  #these attributes are for the opt results
        self.assignments = None
        self.open_centres = None
    
    
    def plot_instance(self):
        '''
        This method plots the optimisation instance - so we see a unit square
        with blue squares for the DCs and red dots for the consumers.
        '''
        
        fig, ax = plt.subplots()
        
        # plot data centres as large blue squares, consumers as red dots

        ax.scatter(self.consumer_locations[:, 0], self.consumer_locations[:, 1],
                    marker='o',      # circle
                    s=20,            
                    c='red',         
                    label='Consumers')
        
        ax.scatter(self.datacentre_locations[:, 0],
                    self.datacentre_locations[:, 1],
                    marker='s',      # square
                    s=120,           
                    c='blue',        
                    label='Data Centres')
        
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
                
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1.02, 1))      
        
        plt.tight_layout()  # make room for the legend
        plt.show()            
        
    def plot_density(self):
        '''
        This method plots the density of the consumers, which can be viewed as
        a 2D density plot for a mixture of bivariate Gaussians.
        
        '''
        kde = gaussian_kde(self.consumer_locations.T)

        # Create evaluation grid
        xmin, ymin = 0, 0
        xmax, ymax = 1, 1
        x_lin = np.linspace(xmin, xmax, 200)
        y_lin = np.linspace(ymin, ymax, 200)
        
        X, Y = np.meshgrid(x_lin, y_lin, indexing='xy')
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions), X.shape)
        
        # Plot density map with sample overlay
        plt.figure()
        plt.imshow(Z, origin='lower', aspect='auto',
                   extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('KDE-based Consumer Density Estimate')
        plt.show()
        
        
    def exact_solve(self, alpha):
        '''
        This method interfaces with Gurobi and solves the instance using MIP
        methods (as implemented by Gurobi, so various cutting planes basically).
        
        Alpha is ther parameter for weighting the QoE / latency penalty.
        
        The function will print information from the Gurobi solver, however
        the class is also updated with a new attribute - optimisation_model -
        which contains all of the information returned from the solver.
        
        '''
        #######
        #Read in data from self
        #######

        m = len(self.datacentre_costs)    # number of data‐centres
        n = len(self.consumer_costs[0])   # number of consumers
        alpha = 1.0  #penalty for objective

        D = self.datacentre_costs           #costs to open
        B = self.datacentre_capacities #centre capacities
        c = self.consumer_costs             #consumer delivery costs
        d = self.consumer_latencies #latencies/distances

        #######
        #Set Variables
        #######

        flexmedia_model = gp.Model("flexmedia_datacentre_placement")
        flexmedia_model.setParam('OutputFlag', 0) #prevents console printing

        # Decision variables
        # y[i] = 1 if we open centre i
        y = flexmedia_model.addVars(m, vtype=GRB.BINARY, name="y")

        # x[i,j] = 1 if consumer j is served by centre i
        x = flexmedia_model.addVars(m, n, vtype=GRB.BINARY, name="x")

        #######
        #Objective Function
        ########

        #quicksum just gives the sum of the arrays as sum() does, but is faster in gurobi
        flexmedia_model.setObjective(
            gp.quicksum(D[i] * y[i] 
                        for i in range(m))
          + gp.quicksum((c[i,j] + alpha * d[i,j]) * x[i,j]
                        for i in range(m) for j in range(n)),
            GRB.MINIMIZE
        )

        ########
        #Add constraints
        #######

        #consumer j must be assigned exactly once:
        flexmedia_model.addConstrs(
            (gp.quicksum(x[i,j] for i in range(m)) == 1
             for j in range(n)),
            name="assign_once"
        )

        #only assign consumers to an open centre:
        flexmedia_model.addConstrs(
            (x[i,j] <= y[i]
             for i in range(m) for j in range(n)),
            name="open_if_assigned"
        )

        #centre i serves at most B[i] consumers if opened:
        flexmedia_model.addConstrs(
            (gp.quicksum(x[i,j] for j in range(n)) <= B[i] * y[i]
             for i in range(m)),
            name="capacity"
        )

        #########
        #Solve
        #########

        flexmedia_model.optimize()
        self.optimisation_model = flexmedia_model
        
        ################
        #Extract assignments and which DCs are open
        ##############

        if flexmedia_model.status == GRB.OPTIMAL:
        #if we have optimal solution then store the assignments and open centres
            self.assignments = {
                i: [j for j in range(n) if x[i, j].x > 0.5]
                for i in range(m)
                if y[i].x > 0.5
            }
            self.open_centres = [i for i in range(m) if y[i].x > 0.5]
        else:
            print("Warning: No optimal solution found.")
            
    def print_exact_solution(self):
        '''
        This method prints out the exact solution should it exist, showing which
        consumers go from each DC. Will be of limited to no use for large
        instances!
        '''
        if self.optimisation_model is None:
            print("Optimisation istance has not been solved exactly yet")
            return
        
        for i in self.open_centres:
            print(f"Data centre {i} opened")
            for j in self.assignments[i]:
                print(f"  - Viewer {j} assigned")
                
    def plot_exact_solution(self):
        '''
        This method plots the exact solution, with each DC a different coloured
        square, and each consumer a dot coloured according to the DC it is 
        assigned to.
        
        To help with plot clarity, this function only shows the opened DCs.
        
        Again, likely to be of limited to no use for large instances !
        '''
        
        if self.optimisation_model is None:
            print("Optimisation istance has not been solved exactly yet")
            return
        
        m = len(self.datacentre_costs)    # number of data‐centres
        n = len(self.consumer_costs[0])   # number of consumers
        
        consumer_to_centre = np.full(n, -1, dtype=int)
        for i, consumers in self.assignments.items():
            for j in consumers:
                consumer_to_centre[j] = i
        
        cmap = plt.get_cmap('tab10', m)
        dc_colors = cmap(np.arange(m))
        
        plt.figure(figsize=(6, 6))
        for i in self.open_centres:
            x, y = self.datacentre_locations[i]
            plt.scatter(x, y, marker='s', s=200, color=dc_colors[i], edgecolor='k', label=f'Data Centre {i}')
        
        for j in range(n):
            i = consumer_to_centre[j]
            x, y = self.consumer_locations[j]
            plt.scatter(x, y, marker='o', s=50, color=dc_colors[i])
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()
        
    def _solve_facility_subproblem(self, i, lambda_vec, D, c, d, alpha, B):
        """
        This function is designed to be used as part of the Lagrangian solver, 
        rather than called directly. 
        
        It solves (using Gurobi) the inner minimisation problem:
            phi_i(lambda) = min_{x_i., y_i} D_i y_i +
                            sum_j (c_ij + alpha d_ij - lambda_j) x_ij
             s.t. x_ij <= y_i, sum_j x_ij <= B_i y_i, x_ij, y_i in {0,1}
        
        The function returns the min objective value, and then x_i and y_ values.
        x_i will be a length n_consumers array of 0/1s, and y_i is a binary 0/1.
        
        For clarity the inputs are:
            - i: the index of the DC
            - lambda_vec: the lambda values, one per consumer
            - D: the cost of opening a datacentre
            - c: the consumer costs for this datacentre.
            - d: the distances of the consumers to this datacentre.
            - alpha: the QoE weight.
            - B: the DC capacities
        
        """
        n = c.shape[1]
        model = gp.Model()
        model.setParam('OutputFlag', 0)
    
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        x = model.addVars(n, vtype=GRB.BINARY, name=f"x_{i}_")
    
        #The solver for the inner min centres around the reduced costs, which are:
        reduced_costs = c[i, :] + alpha * d[i, :] - lambda_vec  
        model.setObjective(D[i] * y + gp.quicksum(reduced_costs[j] * x[j] for j in range(n)), GRB.MINIMIZE)
    
        model.addConstrs(x[j] <= y for j in range(n))
        model.addConstr(gp.quicksum(x[j] for j in range(n)) <= B[i] * y)
    
        model.optimize()
    
        #Extract the current solution (as int) decision variables
        x_i = (np.array([x[j].X for j in range(n)]) > 0.5).astype(int)
        y_i = int(y.X > 0.5)
        return model.objVal, x_i, y_i
    
        
        
    def lagrangian_solve(self, alpha, T=200, theta=1.0, UB=np.inf):
        '''
        This function implements a solver for the flexmedia instance
        by formulating it as a Lagrangian relaxation, solving its dual,
        and then repairing the solution so that it is feasible for the 
        unrelaxed primal.
        
        The approach used to update lambda at each step is based on the 
        subgradient. We compute the step size using the subgradient and then
        use the Polyak method to update the lambda values based on this step
        size.
        '''
        
    
        D = self.datacentre_costs           #costs to open
        B = self.datacentre_capacities #centre capacities
        c = self.consumer_costs             #consumer delivery costs
        d = self.consumer_latencies #latencies/distances
    
        m, n = c.shape
        lambda_vec = np.zeros(n)  #Start with zeroes for lambda 
        #note that the lambda values are free so can be any real
        best_L = -np.inf
        best_iter = None
    
        for t in range(1, T + 1):
            phi_vals = np.zeros(m)
            x_stack = np.zeros((m, n), dtype=int)
            y_open = np.zeros(m, dtype=int)
    
            #Solve each facility subproblems
            for i in range(m):
                phi_i, x_i, y_i = self._solve_facility_subproblem(i, lambda_vec, D,
                                                             c, d, alpha, B)
                phi_vals[i] = phi_i
                x_stack[i, :] = x_i
                y_open[i] = y_i
    
            #Compute the value of the Lagrangian objective
            L_val = lambda_vec.sum() + phi_vals.sum()
            if L_val > best_L:
                best_L = L_val
                best_iter = t
    
            #Subgradient g_j = 1 - sum_i x_ij
            #Compute the subgradient and use it to compute the step size
            g = 1.0 - x_stack.sum(axis=0)
            gnorm2 = float(np.dot(g, g))
    
            #If UB is known and g != 0 use Polyak update; else diminishing
            if np.isfinite(UB) and gnorm2 > 1e-12 and (UB - L_val) > 0:
                step = theta * (UB - L_val) / gnorm2
            else:
                step = theta / np.sqrt(t)
    
            #update lambda vec using the subgradient based step size.
            lambda_vec = lambda_vec + step * g
        
            # Early exit if subgradient is zero (optimal for given UB)
            if gnorm2 < 1e-12:
                break
            
        #####
        #Repair Lagrangian Solution To Obtain Feasible 
        #####
    
    
        #####
        #Update Class
        #####
        
        self.lagrangian_bound = best_L
        self.lambda_vec = lambda_vec
        self.optimisation_model = "Lagrangian"
    
            
        
        
        
        
        
        
        
        
        