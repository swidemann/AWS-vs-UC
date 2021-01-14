#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

from gurobipy import Model, GRB, quicksum
import Printing as prt
import Writing as wrt
import time
from L_Bound import compute_L_anticipativity_integer_no_capacity, compute_L_SSLP 
from Models_master import create_master_FSC_sb, create_master_FSC_sb_ContFlow
from Models_satellite import solve_integer_Qs_SingleSat, create_second_stage_satellites_FSC_SingleSat, update_model_Qs, update_model_Qs_ContFlow
from Improvements_functions import calculate_transf_cost, generate_similar_flights, bounds_for_similar, check_best_bound
import sys
import statistics as st
import time

import math

try:
try:
    instance = sys.argv[1]
    noshow_str = sys.argv[2].upper()
    if noshow_str not in ["NOSHOW", "SHOW"]:
        raise Exception
    noshow = True if noshow_str == "NOSHOW" else False
    var_percentage = int(sys.argv[3])
    n_airports_str = sys.argv[4]
    n_airports = int(n_airports_str[n_airports_str.find("air")+3:])
    consolidate = sys.argv[5].upper()
    if consolidate not in ["DEFAULT", "CONSBIN", "CONT", "CONSCONT", "CONSBINCONT"]:
        raise Exception
    L_method = sys.argv[6]
    if L_method not in ["solveL", "simpleL"]:
        raise Exception
    hours = sys.argv[7]
    if hours == "None":
        master_time = None
    else:
        master_time = float(hours)*60*60
    partial_str = sys.argv[8].upper()
    if partial_str not in ["PARTIAL", "FULL"]:
        raise Exception
    partial = True if partial_str == "PARTIAL" else False
    if partial:
        print(" PARTIAL CUTS")
    compute_Q_outsample_str = sys.argv[9].upper()    
    if compute_Q_outsample_str not in ["QOUT", "NOQOUT"]:
        raise Exception
    compute_Q_outsample = True if compute_Q_outsample_str == "QOUT" else False
except Exception as err:
    formato = "formato: <modelo.py> <instance> <show or noshow> <var: 10,30,50> air<# airports> <consolidate> <Lmethod> <hours> <NS or reset> <satellite: partial or full> <Qout or noQout>"
    print(formato)
    sys.exit()

print("\n*********************Setting instance to {}*********************\n".format(instance))

##############
### Inputs ###
##############

write_sol = False
write_model = False
write_ods = False
plot_base = False
plot_sol = False
write_cargo_routing = False
plot_convergence = False
write_cuts = False


print(f"Single Satelite Reset")
line = "Cuts added: Integer QTbound reactive global 1fv"
print("\n*** {} ***".format(line))
    
models_dir = "FSC Models"
solutions_dir = "FSC Solutions"
images_dir = "FSC Images"
logs_dir = "FSC Logs"
ods_dir = "FSC ODs"

epsilon = 1000 # check theta > E[Q(y tongo)] + epsilon
if L_method == "solveL":
    solve_for_L = True  # solve a DEF SSLP for L, otherwise use Big M approach
elif L_method == "simpleL":
    solve_for_L = False

theta_Q = []  # pairs (theta, Q) to test the prediction 

# plot theta vs Q(y)
theta_int_values = []
Q_int_values = []
int_cut_times = []

###########################
### Generate Parameters ###
###########################

from Generator import generate_parameters, generate_sizebound_parameters, consolidate_cargo, generate_continous_parameters

days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, created_cargo_types, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, S_OUTSAMPLE, size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE = generate_parameters(instance, n_airports, noshow, var_percentage)
Cargo_base, OD_base = Cargo, OD

# Consolidate Cargo
if consolidate == "DEFAULT":
    consolidated_cargo, continuous_cargo, consolidate_continuous_cargo, consolidate = False, False, False, "Default"
elif consolidate == "CONSBIN":
    consolidated_cargo, continuous_cargo, consolidate_continuous_cargo, consolidate = True, False, False, "ConsBin"
elif consolidate == "CONT":
    consolidated_cargo, continuous_cargo, consolidate_continuous_cargo, consolidate = False, True, False, "Cont"
elif consolidate == "CONSCONT":
    raise Exception
elif consolidate == "CONSBINCONT":
    consolidated_cargo, continuous_cargo, consolidate_continuous_cargo, consolidate = True, True, False, "ConsBinCont"
# consolidate Cargo
if consolidated_cargo:
    original_cargo, original_size, original_vol, original_inc, original_OD, original_mand = Cargo, size, vol, inc, OD, mand
    Cargo, size, vol, inc, OD, mand, ex  = consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, consolidate_continuous_cargo=consolidate_continuous_cargo)
# create vol/size and inc/size for continuous Cargo
if continuous_cargo:
    incperkg, volperkg = generate_continous_parameters(Cargo, size, vol, inc, S)


incperkg_sb, volperkg_sb = generate_sizebound_parameters(Cargo, inc, size, vol, S)
last_hour = days*24


########################
### Satellite Models ###
########################
if partial:  # partial gaps
    from Models_satellite import gaps_to_assign
    next_gap_to_solve = {}  # {y0_key: max(lastgap of Q(y0,s) for s in S) }
#integer
# passing s=1 to create only one model
if not continuous_cargo:
    int_sat, int_r1, y_sat, x_sat, w_sat, q_sat, zplus_sat, r_sat, ss6, ss7, ss10, ss11 = create_second_stage_satellites_FSC_SingleSat(days, 1, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, integer=True)
    const_ss = {6: ss6, 7: ss7, 10: ss10, 11: ss11}
else:
    int_sat, int_r1, y_sat, f_sat, zplus_sat, r_sat, ss5_list, ss7 = create_second_stage_satellites_FSC_SingleSat(days, 1, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, integer=True, continuous_cargo=continuous_cargo, volperkg=volperkg, incperkg=incperkg)
    const_ss = {5: ss5_list, 7: ss7}
int_Q_hash = {}  #{y0_key str: Q(y tongo)}
int_Q_bound_hash = {}  #{y0_key str: Q(y tongo) best bound}


############
# QTBounds #
############

int_sol_hash = {}  #{y0_key str: y0_param dict}
QTbound_hash = {}  #{y0_key str: QT bound}
bound_times = []
proactive_bound_times = []


################
### Callback ###
################

def actualizar_sat_r1(int_sat, int_r1, int_sat_nuevo, int_r1_nuevo):
    int_sat = int_sat_nuevo
    int_r1 = int_r1_nuevo


def opt_cut(model, where):
    if where == GRB.Callback.MIPSOL:  # for a given integer first stage solution
        startTime = time.time()
        y0_param_preprocessed = model.cbGetSolution(y0)  #{y[i, j, k]}
        y0_param = {}
        y0_key = ""  # cant hash with a dict type (y0_param), so transform it to a str
        for i,j in A:
            for k in K:
                y0_param[i,j,k] = round(y0_param_preprocessed[i,j,k])
                y0_key += str(y0_param[i,j,k])
        theta_tongo = model.cbGetSolution(theta)
        # calculate FSC(y^)
        FSC = sum(lc[(arco, k)]*y0_param[arco[0], arco[1], k] for arco in AF for k in K)
        # print("FSC: {}, theta tongo {}".format(FSC, theta_tongo))
        if partial:
            if y0_key not in next_gap_to_solve.keys():
                next_gap_to_solve[y0_key] = gaps_to_assign[0]

        # if for this solution Q hasnt been solved
        if (not partial and y0_key not in int_Q_hash.keys()) or (partial and next_gap_to_solve[y0_key] != None):
            # print("New solution, solving QTbound(y)")
            # predict a bound from a previous solution
            QT_bound =  check_best_bound(y0_param, int_sol_hash, int_Q_bound_hash, sc, V, gap, tv, AF, K, S)
            # print("QT_bound: ", QT_bound)
            # hash the QTbound if its for a new sol or is better than the previous
            if y0_key not in QTbound_hash.keys():
                QTbound_hash[y0_key] = QT_bound
            elif QT_bound < QTbound_hash[y0_key]:
                QTbound_hash[y0_key] = QT_bound
            comment = ""
            # if doesnt cut current solution (y, theta), solve Q
            if not theta_tongo > QT_bound + epsilon:
                # print("QTbound didnt cut, solving Q")
                Qs_int_calculated = {}
                Qs_bound_int = {}
                Qs_gaps_int = {}
                for s in S:
                    if not continuous_cargo:
                        ss6, ss7, ss10, ss11 = update_model_Qs(const_ss, int_sat, y_sat, x_sat, w_sat, q_sat, zplus_sat, r_sat, days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv)
                        const_ss[6], const_ss[7], const_ss[10], const_ss[11] = ss6, ss7, ss10, ss11
                    else:
                        ss5_list, ss7 = update_model_Qs_ContFlow(const_ss, int_sat, y_sat, f_sat, zplus_sat, r_sat, days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, volperkg, incperkg)
                        const_ss[5], const_ss[7] = ss5_list, ss7
                    if partial:
                        Qs_int_calculated[s], Qs_bound_int[s], Qs_gaps_int[s] = solve_integer_Qs_SingleSat(y0_param, s, A, K, int_r1, int_sat, partial_gap=next_gap_to_solve[y0_key])
                    else:
                        Qs_int_calculated[s], Qs_bound_int[s] = solve_integer_Qs_SingleSat(y0_param, s, A, K, int_r1, int_sat)
                if partial:
                    Q_av_gaps_int = sum(Qs_gaps_int[s] for s in S)/len(S)
                    if Q_av_gaps_int <= gaps_to_assign[-1]:  # av <= 1% 
                        next_gap_to_solve[y0_key] = None  # finish
                    else:
                        next_gap_to_solve[y0_key] = max(gap for gap in gaps_to_assign if gap < Q_av_gaps_int)  # the next smaller gap
                Q_int_calculated = sum(Qs_int_calculated[s] for s in S)/len(S)
                Q_int_bound = sum(Qs_bound_int[s] for s in S)/len(S)
                # print("Q_int_calculated: ", Q_int_calculated)
                theta_Q.append( (theta_tongo, Q_int_calculated) )
                int_Q_hash[y0_key] = Q_int_calculated
                int_Q_bound_hash[y0_key] = Q_int_bound
                int_sol_hash[y0_key] = y0_param  # to calculate T(y02,y01)
                comment = "QTbound didnt cut; solved Q"
                # adding feasible solution (y tongo, Q(y tongo))
                this_value = Q_int_calculated - FSC
                if this_value > model._current_incumbent:
                    # print("*** solution better than incumbent ***")
                    model._sol_to_add = model.cbGetSolution(model.getVars())
                    model._theta_to_add = Q_int_calculated  
                    model._current_incumbent = this_value
                if Q_int_calculated > L:
                    print("\n########## ERROR: Q(y tongo)={:.0f} > {:.0f}=L ##########\n".format(Q_int_calculated, L))
                if theta_tongo > Q_int_calculated + epsilon:
                    # print(" ##Theta tongo {:.0f} > Q(y tongo) {:.0f}, adding INTEGER optimality cut##".format(theta_tongo, Q_int_calculated))
                    SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] > 0.5]
                    not_SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] < 0.5]   
                    delta_callback = quicksum(1 - y[0][i, j, k] for i,j,k in SUP_y) + quicksum(y[0][i, j, k] for i,j,k in not_SUP_y) 
                    if partial:
                        model.cbLazy(theta <= (L - Q_int_bound)*delta_callback + Q_int_bound)
                    else:
                        model.cbLazy(theta <= (L - Q_int_calculated)*delta_callback + Q_int_calculated)
                    model._Q_cblazy_added += 1
                else:
                    comment = comment + ", Q didnt cut; not added"
                    # print("**** Theta tongo {:.0f} <= Q(y tongo) {:.0f}, NO NEW INTEGER CUT TO ADD ****".format(theta_tongo, Q_int_calculated))
                theta_int_values.append(theta_tongo)
                Q_int_values.append(Q_int_calculated)
                runTime = time.time() - startTime   
                int_cut_times.append(runTime)
                if write_cuts:
                    wrt.write_cuts(logs_dir, model.ModelName, FSC, theta_tongo, Q_int_calculated, runTime, integer=True, QT_bound=False, comment=comment)            

                # GLOBAL CUTS
                proactive_startTime = time.time()
                # for every operated flight
                for i_bar, j_bar in AF:                    
                    for k_bar in K:
                        if y0_param[i_bar, j_bar, k_bar] == 1:
                            # similar flights to the observed one
                            Similar = generate_similar_flights(y0_param, i_bar, j_bar, k_bar, AF, airports, K)
                            if Similar != []:
                                # QTb in this vicinity
                                Q_y2_p = Q_int_calculated
                                T_y2_y1_p = sum(sc[s][(i_bar, j_bar), k_bar] for s in S)/len(S)
                                QT_bound_p = Q_y2_p + T_y2_y1_p
                                #add global cut
                                Omega_1 = [(i,j,k) for i,j in AF for k in K if (y0_param[i, j, k] > 0.5 and i,j not in Similar)]
                                Omega_2 = [(i,j,k) for i,j in AF for k in K if (y0_param[i, j, k] < 0.5 and i,j not in Similar)]   
                                Delta_1 = quicksum(1 - y[0][i, j, k] for i,j,k in Omega_1) + quicksum(y[0][i, j, k] for i,j,k in Omega_2) 
                                Delta_2 = quicksum(y[0][i, j, k] for i,j in Similar for k in K if k != k_bar) + quicksum(y[0][i_bar, j_bar, k] for k in K)
                                a_hat, b_hat = bounds_for_similar(Similar)
                                Delta_3 = 1 - nu_two[a_hat][b_hat, k_bar]
                                # GC constraints
                                for i,j in Similar:
                                    #GC1
                                    model.cbLazy(y[0][i, j, k_bar] <= nu_one[a_hat][b_hat, k_bar])
                                #GC2
                                model.cbLazy(nu_one[a_hat][b_hat, k_bar] <= quicksum(y[0][i, j, k_bar] for i,j in Similar))
                                #GC3
                                model.cbLazy(2*nu_one[a_hat][b_hat, k_bar] - quicksum(y[0][i, j, k_bar] for i,j in Similar) <= nu_two[a_hat][b_hat, k_bar])
                                #global cut
                                model.cbLazy(theta <= QT_bound_p + (Delta_1 + Delta_2 + Delta_3)*L)
                                model._QTbound_global_cuts += 1                            
                proactive_runTime = time.time() - proactive_startTime
                proactive_bound_times.append(proactive_runTime)

            # if it cuts the current solution, add the QTbound
            else:
                SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] > 0.5]
                not_SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] < 0.5]   
                delta_callback = quicksum(1 - y[0][i, j, k] for i,j,k in SUP_y) + quicksum(y[0][i, j, k] for i,j,k in not_SUP_y) 
                model.cbLazy(theta <= (L - QT_bound)*delta_callback + QT_bound)
                model._QTbound_cblazy_added += 1

                runTime_bound = time.time() - startTime
                bound_times.append(runTime_bound)
                if write_cuts:
                    wrt.write_cuts(logs_dir, model.ModelName, FSC, theta_tongo, QT_bound, runTime_bound, integer=True, QT_bound=True, comment=comment) 

        elif (not partial and y0_key in int_Q_hash.keys()) or (partial and next_gap_to_solve[y0_key] == None):
            Q_int_calculated = int_Q_hash[y0_key]  # for plotting
            if theta_tongo > Q_int_calculated + epsilon:  # add the cut if the first theta wasnt the optimal for the schedule
                SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] > 0.5]
                not_SUP_y = [(i,j,k) for i,j in AF for k in K if y0_param[i, j, k] < 0.5]   
                delta_callback = quicksum(1 - y[0][i, j, k] for i,j,k in SUP_y) + quicksum(y[0][i, j, k] for i,j,k in not_SUP_y) 
                model.cbLazy(theta <= (L - Q_int_calculated)*delta_callback + Q_int_calculated)
                model._Q_cblazy_added += 1

            theta_int_values.append(theta_tongo)
            Q_int_values.append(Q_int_calculated)
            # print("***** VISITED SOLUTION TWICE, NOTHING TO SOLVE. Q(y) = {:.0f} *****".format(Q_int_calculated))
    # adding the feasible solution as the incumbent for the node
    elif where == GRB.Callback.MIPNODE:
        if len(model._sol_to_add) > 0:
            # print("*** ADDING FEASIBLE SOLUTION ***")
            model.cbSetSolution(model.getVars(), model._sol_to_add)
            model.cbSetSolution(theta, model._theta_to_add)
            model._sol_to_add.clear()
            model._theta_to_add = 0



# Compute L
#bigM approach
L_bigM = round( max(sum(inc[s][item] for item in Cargo) for s in S) )
if solve_for_L:
    #solving approach
    Lval, Lbound, Lstatus = compute_L_anticipativity_integer_no_capacity(instance, days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, write_log=False)
    # Lval, Lbound, Lstatus = compute_L_SSLP(instance, days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, nfc, dfc, delta, V, gap, tv, L_time, L_gap, write_log=False)
    L = min(L_bigM, Lbound)  # choose the best method
    if L == L_bigM:
        L_method += " but bigM was better"
else:
    L = L_bigM
print("\n L method: {}, L value: {}".format(L_method, L))      


####################
### Master Model ###
####################

if partial:
    model_name = 'FSC sb QTb global 1-fv SingleSat partial {} i-{}'.format(L_method, instance)
else:
    model_name = 'FSC sb QTb global 1-fv SingleSat {} i-{}'.format(L_method, instance)
model_name = n_airports_str + " " + consolidate + " " + model_name

if not continuous_cargo:
    model, log_name, y, y0, theta, FSC = create_master_FSC_sb(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, model_name, instance, L, L_method, master_time, volperkg_sb, incperkg_sb)
else:
    model, log_name, y, y0, theta, FSC = create_master_FSC_sb_ContFlow(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, model_name, instance, L, L_method, master_time, volperkg_sb, incperkg_sb)

#global cut
model._QTbound_global_cuts = 0
nu_one = {}
nu_two = {}
for n in N:
    latter_nodes = [j for i,j in AF if i[:3] == n[:3] and int(j[3:]) > int(n[3:])]
    nu_one[n] = model.addVars(latter_nodes, K, vtype=GRB.INTEGER, name='nu_1_{}'.format(n))
    nu_two[n] = model.addVars(latter_nodes, K, vtype=GRB.INTEGER, name='nu_2_{}'.format(n))
model.update()


if write_model:
    wrt.write_model(models_dir, "model {} .lp".format(model.ModelName), model)
    # wrt.write_model(models_dir, "{} model i-{}.mps".format(model.ModelName, instance), model)
if write_ods:
    wrt.write_ods(ods_dir, "ODs i-{}.txt".format(instance), Cargo, OD, S, size, created_cargo_types, inc)
if write_cuts:
    wrt.clear_cuts(logs_dir, model.ModelName)

#solve model    
print("\n###### SOLVING {} ######\n".format(model.ModelName))
model.optimize(opt_cut)

incumbent_found = True
try:
    print("\nFSC : {:.2f}".format(FSC.getValue()))
    print("theta : {:.2f}".format(theta.x))
    print("Overall = -FSC + theta : {:.2f}".format(model.objVal))
except:
    print(" ### NO INCUMBENT, CALCULATING MANUALLY ### ")
    incumbent_found = False
    max_y0_key = max(int_Q_hash, key=int_Q_hash.get)
    y0_param = int_sol_hash[max_y0_key]
    FSC_manual = sum(lc[(arco, k)]*y0_param[arco[0], arco[1], k] for arco in AF for k in K)
    theta_manual = int_Q_hash[max_y0_key]
    overall_manual = theta_manual - FSC_manual
    gap_manual = round( (model.ObjBound-overall_manual)/overall_manual*100, 4 )
    print("Best objective {:.12e}, best bound {:.12e}, gap {:.4f}%".format(overall_manual, model.ObjBound, gap_manual))
    print("\nFSC : {:.2f}".format(FSC_manual))
    print("theta : {:.2f}".format(theta_manual))
    print("Overall = -FSC + theta : {:.2f}".format(overall_manual))

if len(int_cut_times) > 0:
    av_int = sum(int_cut_times)/len(int_cut_times)
else:
    av_int = 0
print("\nInteger Benders Cuts computed: {}, added: {}, total time: {:.4f} , average time: {:.4f}".format(len(int_cut_times), model._Q_cblazy_added, sum(int_cut_times), av_int))
if len(bound_times) > 0:
    av_bound_time = sum(bound_times)/len(bound_times)
else:
    av_bound_time = 0
print("\nReactive QT Bounds computed: {}, added: {}, total time: {:.4f} , average time: {:.4f}".format(len(int_cut_times)+len(bound_times), model._QTbound_cblazy_added, sum(bound_times), av_bound_time))    
if len(proactive_bound_times) > 0:
    av_proactive_bound_time = sum(proactive_bound_times)/len(proactive_bound_times)
else:
    av_proactive_bound_time = 0
print("\nGlobal QT Bounds computed: {}, added: {}, total time: {:.4f} , average time per iteration: {:.4f}".format(model._QTbound_global_cuts, model._QTbound_global_cuts, sum(proactive_bound_times), av_proactive_bound_time))    


print("\n *********** PREDICCION DE Q: theta vs Q ************\n ")
difs = [(theta-Q)/Q*100 for theta,Q in theta_Q if theta >= Q]
if len(difs) > 0:
    print("En promedio theta es {:.2f}% mayor que el valor de Q ((theta-Q)/Q*100)".format(st.mean(difs)))
    if len(difs) > 1:
        print("Desviación estándar: {:.2f}%".format(st.stdev(difs)))
else:
    print("Puntos de data insuficientes")


######################################
### SCHEDULE OUTSAMPLE PERFORMANCE ###
######################################
if compute_Q_outsample:
    from Models_satellite import solve_integer_Qs_SingleSat, create_second_stage_satellites_FSC_SingleSat, update_model_Qs

    gap_outsample = 0.02
    print()
    print("\n###################### SCHEDULE OUTSAMPLE PERFORMANCE ######################")
    print(f"\nComputing performance of the flight schedule over {len(S_OUTSAMPLE)} outsample second stage scenarios gap {gap_outsample}...")

    performance_StartTime = time.time()
    # preprocess first stage schedule and test second stage value
    if incumbent_found:
        max_y0_key = ""
        y0_param = {}
        for i,j in A:
            for k in K:
                y0_param[i,j,k] = round(y[0][i, j, k].x)
                max_y0_key += str(y0_param[i,j,k])
    print(f"Vector y: {max_y0_key}")

    #generate SingleSat
    int_sat, int_r1, y_sat, x_sat, w_sat, q_sat, zplus_sat, r_sat, ss6, ss7, ss10, ss11 = create_second_stage_satellites_FSC_SingleSat(days, 1, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo_base, OD_base, size_OUTSAMPLE, vol_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, cv, cf, ch, inc_OUTSAMPLE, lc, sc_OUTSAMPLE, delta, V, gap, tv, integer=True)
    const_ss = {6: ss6, 7: ss7, 10: ss10, 11: ss11}

    #solve
    Qs_bound_int = {}
    Qs_int_calculated = {}
    times_list = []
    Q_list = []
    for s in S_OUTSAMPLE:
        this_StarTime = time.time()
        ss6, ss7, ss10, ss11 = update_model_Qs(const_ss, int_sat, y_sat, x_sat, w_sat, q_sat, zplus_sat, r_sat, days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo_base, OD_base, size_OUTSAMPLE, vol_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, cv, cf, ch, inc_OUTSAMPLE, lc, sc_OUTSAMPLE, delta, V, gap, tv)
        const_ss[6], const_ss[7], const_ss[10], const_ss[11] = ss6, ss7, ss10, ss11
        Qs_int_calculated[s], Qs_bound_int[s], _ = solve_integer_Qs_SingleSat(y0_param, s, A, K, int_r1, int_sat, partial_gap=gap_outsample)
        this_RunTime = round( time.time() - this_StarTime, 2)
        times_list.append(this_RunTime)
        Q_list.append(round(Qs_int_calculated[s], 2))
        print(" Scenario OUTSAMPLE {}: Q = {:.2f}, time = {}".format(s, Qs_int_calculated[s], this_RunTime))
    Q_int_calculated = sum(Qs_int_calculated[s] for s in S_OUTSAMPLE)/len(S_OUTSAMPLE)
    performance_RunTime = round( time.time() - performance_StartTime, 2)
    sd = round(st.stdev(Q_list), 2)
    print(f"Q mean : {st.mean(Q_list)}, min: {min(Q_list)}, max: {max(Q_list)}, SD: {sd}, Percentage Variation: {round(sd/st.mean(Q_list)*100, 2)}%")
    print(f"Real second stage value for the schedule: {round(Q_int_calculated, 2)}")
    print(f"Real overall = -FSC + sum(Q_s for s in S_OUTSAMPLE)/|S_OUTSAMPLE|: {round(-FSC.getValue()+Q_int_calculated, 2)}")
    print(f"Time: {performance_RunTime}, mean : {st.mean(times_list)}, min: {min(times_list)}, max: {max(times_list)}")


status = model.status
if status != GRB.Status.INFEASIBLE:
    #PRINT SOLUTION
    if write_sol:
        wrt.write_model(solutions_dir, "solution {}.sol".format(model.ModelName), model)
        # wrt.write_model(soltions_dir, "{} solution i-{}.mst".format(model.ModelName, instance), model)
    wrt.write_log(logs_dir, log_name)

    #plot solution
    last_hour = days*24
    if plot_base:
        print( "\n########### SPACE-TIME NETWORK ##########")
        prt.print_original_network(airports, N, A, last_hour, instance, images_dir)
    if plot_sol:
        print( "\n########### FIRST STAGE SCHEDULE ##########")
        for k in K:
            prt.print_aggregated_fs_aircraft_schedule(airports, N, A, last_hour, k, n, y, instance, images_dir)

        # ## TODO mover al final del callback, cuando theta <= theta calculado
        # print( "\n########### SECOND STAGE SCHEDULE ##########")
        # for s in S:
        #     print( "###### Scenario {} #####".format(s))
        #     for k in K:
        #         prt.print_aggregated_ss_aircraft_schedule(airports, N, A, last_hour, k, n, y, s, instance, images_dir)

    if plot_convergence:
        prt.print_convergence(theta_int_values, Q_int_values, model.ModelName, images_dir, instance, integer=True)
   
else:
    print('Optimization was stopped with status {}'.format(status))
    # compute Irreducible Inconsistent Subsystem
    model.computeIIS()
    for constr in model.getConstrs():
        if constr.IISConstr:
            print('Infeasible constraint: {}'.format(constr.constrName))
