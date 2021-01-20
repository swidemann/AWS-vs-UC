#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

from gurobipy import Model, GRB, quicksum
import Printing as prt
import Writing as wrt
import sys
import statistics as st
import time
import os

instance = sys.argv[1]
noshow_str = sys.argv[2].upper()
if noshow_str not in ["NOSHOW", "SHOW"]:
    raise Exception
noshow = True if noshow_str == "NOSHOW" else False
var_percentage = int(sys.argv[3])
n_airports_str = sys.argv[4]
n_airports = int(n_airports_str[n_airports_str.find("air")+3:])


print("\n*********************Setting instance to {}*********************\n".format(instance))

###########################
### Generate Parameters ###
###########################
from Generator import generate_parameters           

days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, created_cargo_types, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, S_OUTSAMPLE, size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE = generate_parameters(instance, n_airports, noshow, var_percentage)
Cargo_base, OD_base = Cargo, OD

from L-shaped_FSC_hashing import y as y_param

write_sol = False
write_model = False
write_ods = False
plot_base = False
plot_sol = False

models_dir = "Aircraft Routing Models"
solutions_dir = "Aircraft Routing Solutions"
images_dir = "Aircraft Routing Images"
logs_dir = "Aircraft Routing Logs"
ods_dir = "Aircraft Routing ODs"

#############
### Model ###
#############

model = Model('Aggregated Aircraft Routing {}'.format(instance))
log_name = "log {}.log".format(model.ModelName)
model.setParam("LogFile", log_name)


#################
### Variables ###
#################

#Fleet Scheduling parameters
y = {}  #{s: y[i, j, k]}
for s in S+[0]:
    ys = {}
    for k in K:
        for i,j in A:
            ys[i, j, k] = round(y_param[s][i, j, k].x)
    y[s] = ys

#Aircraft Routing variables
h = {}  #{s: {k: h[arco, m]}}
for s in S+[0]:
    ys = {}
    for k in K:
        ys[k] = model.addVars(A, n[k], vtype=GRB.BINARY, name='h{}{}'.format(k, s))
    h[s] = ys


###################
### Constraints ###
###################

#1
model.addConstrs((quicksum(h[s][k][i, j, m] for m in n[k]) 
        == y[s][i, j, k] for k in K for i,j in A for s in S+[0]), name='(1)')
#2
model.addConstrs((quicksum(h[s][k][i, j, m] for i,j in A if i == ii) 
        == av[(ii, k, m)] for k in K for m in n[k] for ii in Nf for s in S+[0]), name='(2)')
#3
model.addConstrs((quicksum(h[s][k][i, j, m] for i,j in A if j == jj) - quicksum(h[s][k][i, j, m] for i,j in A if i == jj) 
        == 0 for k in K for m in n[k] for jj in Nint for s in S+[0]), name='(3)')


##########################
### Objective Function ###
##########################

obj = 0
model.setObjective(obj, GRB.MAXIMIZE)
model.Params.Threads = 1
# model.setParam('MIPGap', 0.05)
#model.setParam('TimeLimit', 2*60*60)
model.update()


if write_model:
    wrt.write_model(models_dir, "model {} .lp".format(model.ModelName), model)
    # wrt.write_model(models_dir, "{} model i-{}.mps".format(model.ModelName, instance), model)
if write_ods:
    bool_consolidated = True if consolidate != "DEFAULT" else False
    wrt.write_ods(ods_dir, "ODs i-{}.txt".format(instance), Cargo, OD, S, size, created_cargo_types, inc)

#solve model    
print("\n###### SOLVING {} ######\n".format(model.ModelName))
model.optimize()


status = model.status
if status == GRB.Status.OPTIMAL:
    #PRINT SOLUTION
    if write_sol:
        # wrt.write_parameters(instance, days, S, airports, K, n, OD, N, A, AF, models_dir, "Parameters {} i-{}.txt".format(model.ModelName, instance), model, q, size, RR=True)
    wrt.write_log(logs_dir, log_name)

    #plot solution
    last_hour = days*24
    if plot_base:
        pass

    if plot_sol:
        print( "\n########### SPECIFIC SCHEDULE ##########")
        for s in S+[0]:
            for k in K:
                for m in n[k]:
                    print( "###### Scenario {}, aircraft {} type {} #####".format(s, m, k))
                    prt.print_aggregated_aircraft_routing_specific_schedule(airports, N, A, last_hour, k, m, h, s, instance, images_dir)
else:
    print('Optimization was stopped with status {}'.format(status))
    # compute Irreducible Inconsistent Subsystem
    model.computeIIS()
    for constr in model.getConstrs():
        if constr.IISConstr:
            print('Infeasible constraint: {}'.format(constr.constrName))

### MAXIMUM MEMORY USAGE
os.system(f"grep VmPeak /proc/{os.getpid()}/status")