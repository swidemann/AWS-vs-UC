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
    hours = sys.argv[6]
    if hours == "None":
        master_time = None
    else:
        master_time = float(hours)*60*60
    compute_Q_outsample_str = sys.argv[7].upper()    
    if compute_Q_outsample_str not in ["QOUT", "NOQOUT"]:
        raise Exception
    compute_Q_outsample = True if compute_Q_outsample_str == "QOUT" else False
except Exception as err:
    formato = "formato: <modelo.py> <instance> <show or noshow> <var: 10,30,50> air<# airports> <consolidate> <hours> <Qout or noQout>"
    print(formato)
    sys.exit()

print("\n*********************Setting instance to {}*********************\n".format(instance))

##############
### Inputs ###
##############

write_sol = False
write_model = False
write_ods = True
print_histogram_ods = False
plot_base = False
plot_sol = False
write_cargo_routing = False
write_retimings = False


models_dir = "DEF Models"
solutions_dir = "DEF Solutions"
images_dir = "DEF Images"
logs_dir = "DEF Logs"
time_dir = "DEF RunTimes"
ods_dir = "DEF ODs"
retimings_dir = "DEF Retimings"

master_gap = 0.01
if hours == "None":
    master_time = None
else:
    master_time = float(hours)*60*60
MIPFocus = None  # 1: feasible solutions, 2: optimality, 3: bound
NodefileStart = 0.5  # write nodes to disk after x GB, recomended 0.5


###########################
### Generate Parameters ###
###########################

from Generator import generate_parameters, consolidate_cargo, generate_continous_parameters          

days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, created_cargo_types, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, S_OUTSAMPLE, size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE = generate_parameters(instance, n_airports, noshow, var_percentage)
Cargo_base, OD_base = Cargo, OD
L_bigM = round( max(sum(inc[s][item] for item in Cargo) for s in S) )
print(f"TEST DEF L: {L_bigM}\n")

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
    Cargo, size, vol, inc, OD, mand, ex  = consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, consolidate_continuous_cargo=consolidate_continuous_cargo)
    L_bigM = round( max(sum(inc[s][item] for item in Cargo) for s in S) )
    print(f"TEST DEF CONS_L: {L_bigM}\n")

# create vol/size and inc/size for continuous Cargo
if continuous_cargo:
    incperkg, volperkg = generate_continous_parameters(Cargo, size, vol, inc, S)
    L_bigM = round( max(sum(incperkg[s][item]*size[s][item] for item in Cargo) for s in S) )
    print(f"TEST DEF CONT_L: {L_bigM}\n")

# TESTEANDO ERROR EN ConsCont
for s in S:
    for item in Cargo:
        if ex[s][item] == 0:
            if size[s][item] != 0 or vol[s][item] != 0 or inc[s][item] != 0:
                print(f"ERROR: s {s} item {item} ex {ex[s][item]} size {size[s][item]} vol {vol[s][item]} inc {inc[s][item]}")
        elif continuous_cargo:
            if abs(round(vol[s][item]) - round(volperkg[s][item]*size[s][item])) > 2 or abs(round(inc[s][item]) - round(incperkg[s][item]*size[s][item])) > 2:
                print(f"ERROR: s {s} item {item} ex {ex[s][item]} ")
                print(f"vol {round(vol[s][item])}, volperkg*size {round(volperkg[s][item]*size[s][item])}")
                print(f"inc {round(inc[s][item])}, incperkg*size {round(incperkg[s][item]*size[s][item])}")


#############
### Model ###
#############
model_name = 'DEF i-{}'.format(instance)
model_name = n_airports_str + " " + consolidate + " " + model_name

model = Model(model_name)
log_name = "log {}.log".format(model.ModelName)
model.setParam("LogFile", log_name)


#################
### Variables ###
#################

# first stage
y = {}  #{s: y[i, j, k]}
y0 = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0'))
y[0] = y0
# second stage
if not continuous_cargo:
    x = {}  #{s: {x[i, j, od]}}
    q = {}  #{s: {q[od]}
    w = {}  #{s: {w[i, j, od, k]}}
    for s in S:
        x[s] = model.addVars(A, Cargo, vtype=GRB.BINARY, name='x{}'.format(s))
        q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
else:
    f = {}  #{s: {f[i, j, od]}}
    for s in S:
        f[s] = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, lb=0.0, name='f{}'.format(s))
zplus = {}  #{s: zplus[i, j, k]}
zminus = {}  #{s: zminus[i, j, k]}
for s in S:
    y[s] = model.addVars(A, K, vtype=GRB.INTEGER, name=('y{}'.format(s)))
    zplus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus{}'.format(s))
    zminus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus{}'.format(s))
#retiming
r = {}  # {s: {arco: r[arco2, k]}}
for s in S:
    rs = {}
    for arco in AF:
        rs[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
    r[s] = rs


###################
### Constraints ###
###################

## FIRST STAGE CONSTRAINTS ##
#FS1
model.addConstrs((quicksum(y[0][i, j, k] for k in K) 
        <= 1 for i, j in AF), name='(FS1)')
#FS2                
model.addConstrs((quicksum(y[0][i, j, k] for i,j in A if i == ii) 
        == nav[ii, k] for k in K for ii in Nf), name='(FS2)')
#FS3                
model.addConstrs((quicksum(y[0][i, j, k] for i,j in A if j == jj) - quicksum(y[0][i, j, k] for i,j in A if i == jj) 
        == 0 for k in K for jj in Nint), name='(FS3)')
## SECOND STAGE CONSTRAINTS ##
#SS1
model.addConstrs((quicksum(y[s][i, j, k] for k in K) 
        <= 1 for i,j in AF for s in S), name='(SS1)')
#SS2        
model.addConstrs((y[s][i, j, k] == y[0][i, j, k]
        + zplus[s][i, j, k] - zminus[s][i, j, k] for k in K for i,j in A for s in S), name="(SS2)")
#SS3
model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if i == ii) 
        == nav[ii, k] for k in K for ii in Nf for s in S), name='(SS3)')        
#SS4
model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if j == jj) - quicksum(y[s][i, j, k] for i,j in A if i == jj) 
        == 0 for k in K for jj in Nint for s in S), name='(SS4)')
#SS5
if not continuous_cargo:
    for jj in N:
        for item in Cargo:
            od = OD[item]
            if od[1] == jj:
                #SS5.1
                model.addConstrs((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                    == q[s][item] for s in S), name='(SS5.1)')
            elif od[0] == jj:
                #SS5.2
                model.addConstrs((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                    == -q[s][item] for s in S), name='(SS5.2)')
            else:
                #SS5.3
                model.addConstrs((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                    == 0 for s in S), name='(SS5.3)')
else:
    for jj in N:
        for item in Cargo:
            od = OD[item]
            if mand[0][item] == 1:
                if od[1] == jj:
                    #SS5.1
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        == size[s][item] for s in S), name='(SS5.1)')
                elif od[0] == jj:
                    #SS5.2
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        == -size[s][item] for s in S), name='(SS5.2)')
                else:
                    #SS5.3
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        == 0 for s in S), name='(SS5.3)')
            else:
                if od[1] == jj:
                    #SS5.1
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        <= size[s][item] for s in S), name='(SS5.1)')
                elif od[0] == jj:
                    #SS5.2
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        >= -size[s][item] for s in S), name='(SS5.2)')
                else:
                    #SS5.3
                    model.addConstrs((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                        == 0 for s in S), name='(SS5.3)')
#SS6
if not continuous_cargo:
    model.addConstrs((quicksum(size[s][item]*x[s][i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS6)")
else:
    model.addConstrs((quicksum(f[s][i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS6)")
#SS7
if not continuous_cargo:
    model.addConstrs((quicksum(vol[s][item]*x[s][i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS7)")
else:
    model.addConstrs((quicksum(volperkg[s][item]*f[s][i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS7)")
#SS8
if not continuous_cargo:
    model.addConstrs((x[s][i, j, item] 
            <= quicksum(y[s][i, j, k] for k in K) for item in Cargo for i,j in AF for s in S), name="(SS8)")
else:
    model.addConstrs((f[s][i, j, item]
            <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for item in Cargo for i,j in AF for s in S), name="(SS8)")

if not continuous_cargo:
    #SS9 
    model.addConstrs((x[s][i, j, item] + y[s][i, j, k] 
            <= w[s][i, j, item, k] + 1 for k in K for item in Cargo for i,j in AF for s in S), name="(SS9)")        
    #SS10
    model.addConstrs((q[s][item] 
            >= mand[s][item] for item in Cargo for s in S), name="(SS10)")
    #SS11
    model.addConstrs((q[s][item] 
            <= ex[s][item] for item in Cargo for s in S), name="(SS11)")
#SS12
model.addConstrs((zminus[s][i, j, k] 
        <= y[0][i, j, k] for k in K for i,j in AF for s in S), name="(SS12)")  
#SS13
model.addConstrs((y[0][i, j, k] + zplus[s][i, j, k] 
        <= 1 for k in K for i,j in AF for s in S), name="(SS13)")                 
#z+_bin_flight
model.addConstrs((zplus[s][i, j, k] 
        <= 1 for k in K for i,j in AF for s in S), name="(z+_bin_flight)")
#z-_bin_flight
model.addConstrs((zminus[s][i, j, k] 
        <= 1 for k in K for i,j in AF for s in S), name="(z-_bin_flight)")        
# Retiming       
#RT1
model.addConstrs((y[0][i, j, k] + zminus[s][i, j, k] + zplus[s][i2, j2, k]
        <= 2 + r[s][i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j] for s in S), name="(RT1)")		        
#RT2
model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
        <= y[0][i, j, k] for k in K for i,j in AF for s in S), name="(RT2)")		        
#RT3
model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
        <= zminus[s][i, j, k] for k in K for i,j in AF for s in S), name="(RT3)")      
#RT4
model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i,j in V[i2, j2])
        <= zplus[s][i2, j2, k] for k in K for i2,j2 in AF for s in S), name="(RT4)") 


##########################
### Objective Function ###
##########################

# Definition of the first stage costs
FSC = quicksum(lc[(arco, k)]*y[0][arco[0], arco[1], k] for arco in AF for k in K) 
# Definition of the second stage value        
SSP = {}  #{s: SSPs} second stage profit for each scenario
TRC = {}  #{s: TRCs} total recovery cost
RC = {}  #{s: RCs} re-timing costs
for s in S:
    if not continuous_cargo:
        SSP[s] = (  quicksum(inc[s][item]*q[s][item] for item in Cargo)
                - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, k)]*size[s][item]*w[s][arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                - quicksum(ch[arco]*size[s][item]*x[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    else:
        SSP[s] = (  quicksum(incperkg[s][item]*quicksum(f[s][i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
                - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(min(cv[(arco, k)] for k in K)*f[s][arco[0], arco[1], item] for item in Cargo for arco in AF)  # cv[k1] = cv[k2]
                - quicksum(ch[arco]*f[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    TRC[s] = quicksum( sc[s][(arco, k)]*zplus[s][arco[0], arco[1], k] for k in K for arco in AF)

    RC[s] = quicksum(tv*gap[arco, arco2]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k]
                    - sc[s][arco2, k]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

Q = quicksum(SSP[s] - TRC[s] - RC[s] for s in S)/len(S)

obj = -FSC + Q
model.setObjective(obj, GRB.MAXIMIZE)
model.Params.Threads = 1
if master_gap != None:
    model.setParam('MIPGap', master_gap)
if master_time != None:
    model.setParam('TimeLimit', master_time)
if MIPFocus != None:
    model.setParam('MIPFocus', MIPFocus)
if NodefileStart != None:
    model.setParam('NodefileStart', NodefileStart)
model.update()


if write_model:
    wrt.write_model(models_dir, "model {} .lp".format(model.ModelName), model)
    # wrt.write_model(models_dir, "{} model i-{}.mps".format(model.ModelName, instance), model)
if write_ods:
    wrt.write_ods(ods_dir, "ODs i-{}.txt".format(instance), Cargo, OD, S, size, created_cargo_types, inc)
if print_histogram_ods:
    prt.print_histograms(ods_dir, size, S, Cargo, created_cargo_types, instance)

#solve model    
print("\n###### SOLVING {} ######\n".format(model.ModelName))
model.optimize()

try:
    print("FSC = {}".format(FSC.getValue()))
    print("Q = quicksum(SSP[s] - TRC[s] - RC[s] for s in S)/len(S): {}".format(Q.getValue()))
    print("Overall = -FSC + Q = {}\n".format(model.objVal))

    if True:
        print("vuelos en primera etapa: ", sum(y[0][i, j, k].x for k in K for i,j in AF))
        for s in S:
            print("Scenario ", s)
            if not continuous_cargo:
                transp_cargo = sum(q[s][item].x for item in Cargo)
                print(" Transported Cargo: {:.0f} out of {} ({:.2f}%)".format(transp_cargo, len(Cargo), transp_cargo/len(Cargo)*100))
            else:
                transp_cargo = sum(f[s][i, j, item].x for item in Cargo for i,j in A if j == OD[item][1])
                total_cargo = sum(size[s][item] for item in Cargo)
                print(" Transported Cargo: {:.0f} out of {} ({:.2f}%)".format(transp_cargo, total_cargo, transp_cargo/total_cargo*100))
            ssp, trc, rc  = SSP[s].getValue(), TRC[s].getValue(), RC[s].getValue()
            valor = ssp - trc - rc
            print(" Q[s] = SSP[s] - TRC[s] - RC[s] = {:.0f} - {:.0f} - {:.0f} = {:.0f}".format(ssp, trc, rc, valor))
            print(" vuelos en descartados en segunda etapa: ", sum(zminus[s][i, j, k].x for i,j in AF for k in K))
            print(" vuelos en agregados en segunda etapa: ", sum(zplus[s][i, j, k].x for i,j in AF for k in K))
            print(" Vuelos: ")
            for i,j in AF:
                for k in K:
                    if round(y[s][i, j, k].x) == 1:  # flight exists
                        if not continuous_cargo:
                            transported = sum(size[s][item]*x[s][i, j, item].x for item in Cargo)
                        else:
                            transported = sum(f[s][i, j, item].x for item in Cargo)
                        cargo_load_factor = transported/air_cap[k]
                        print("  {}->{}, {:.0f} kgs transported, Cargo load factor: {:.2f}".format(i, j, transported, cargo_load_factor))
except Exception as err:
    print(err)


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
        # wrt.write_parameters(instance, days, S, airports, K, n, OD, N, A, AF, models_dir, "Parameters {} i-{}.txt".format(model.ModelName, instance), model, q, size, RR=True)
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

        print( "\n########### SECOND STAGE SCHEDULE ##########")
        for s in S:
            print( "###### Scenario {} #####".format(s))
            for k in K:
                prt.print_aggregated_ss_aircraft_schedule(airports, N, A, last_hour, k, n, y, s, instance, images_dir)

    if write_cargo_routing:
        wrt.write_cargo_schedule(ods_dir, "Schedule ODs i-{}.txt".format(instance), A, S, Cargo, OD, ex, x)

    if write_retimings:
        wrt.write_retimings(retimings_dir, "Retimings i-{}.txt".format(instance), S, AF, V, K, y, r, zplus, zminus)
   
else:
    print('Optimization was stopped with status {}'.format(status))
    # compute Irreducible Inconsistent Subsystem
    model.computeIIS()
    for constr in model.getConstrs():
        if constr.IISConstr:
            print('Infeasible constraint: {}'.format(constr.constrName))
