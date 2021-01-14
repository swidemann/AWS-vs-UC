#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

from gurobipy import Model, GRB, quicksum

sat_gap = 0.01
gaps_to_assign = [0.2, 0.1, 0.01]


#################
# UPDATE MODELS #
#################

def update_model_Qs_ContFlow(const_ss, model, y_s, f_s, zplus_s, r_s, days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, volperkg, incperkg):
    # Reset the model to an unsolved state, discarding any previously computed solution information.
    model.reset(1)
    # eliminar las restricciones 
    for const5 in const_ss[5]:
        model.remove(const5)
    model.remove(const_ss[7])
    # crearlas nuevamente
    #SS5
    ss5_list = []
    for jj in N:
        for item in Cargo:
            od = OD[item]
            if mand[0][item]:
                if od[1] == jj:
                    #SS5.1
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        == size[s][item]), name='(SS5.1)')
                elif od[0] == jj:
                    #SS5.2
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        == -size[s][item]), name='(SS5.2)')
                else:
                    #SS5.3
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(SS5.3)')
            else:
                if od[1] == jj:
                    #SS5.1
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        <= size[s][item]), name='(SS5.1)')
                elif od[0] == jj:
                    #SS5.2
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        >= -size[s][item]), name='(SS5.2)')
                else:
                    #SS5.3
                    ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(SS5.3)')
            ss5_list.append(ss5)
    #SS7
    ss7 = model.addConstrs((quicksum(volperkg[s][item]*f_s[i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS7)")

    # Definition of the second stage value        
    SSP_s = (  quicksum(incperkg[s][item]*quicksum(f_s[i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
            - quicksum(cf[(arco, k)]*y_s[arco[0], arco[1], k] for arco in AF for k in K)
            - quicksum(cv[(arco, 1)]*f_s[arco[0], arco[1], item] for item in Cargo for arco in AF)
            - quicksum(ch[arco]*f_s[arco[0], arco[1], item] for item in Cargo for arco in AG) 
            )
    TRC_s = quicksum( sc[s][(arco, k)]*zplus_s[arco[0], arco[1], k] for k in K for arco in AF)

    RC_s = quicksum(tv*gap[arco, arco2]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k]
                    - sc[s][arco2, k]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

    obj = SSP_s - TRC_s - RC_s
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()

    return ss5_list, ss7





def update_model_Qs(const_ss, model, y_s, x_s, w_s, q_s, zplus_s, r_s, days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv):
    # Reset the model to an unsolved state, discarding any previously computed solution information.
    model.reset(1)
    # eliminar las restricciones 
    model.remove(const_ss[6])
    model.remove(const_ss[7])
    model.remove(const_ss[10])
    model.remove(const_ss[11])
    # crearlas nuevamente
    #SS6
    ss6 = model.addConstrs((quicksum(size[s][item]*x_s[i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS6)")
    #SS7
    ss7 = model.addConstrs((quicksum(vol[s][item]*x_s[i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS7)")
    #SS10
    # force mandatory Cargo only when not limited recovery actions
    ss10 = model.addConstrs((q_s[item] 
            >= mand[s][item] for item in Cargo ), name="(SS10)")
    #SS11
    ss11 = model.addConstrs((q_s[item] 
            <= ex[s][item] for item in Cargo ), name="(SS11)")

    # Definition of the second stage value        
    SSP_s = (  quicksum(inc[s][item]*q_s[item] for item in Cargo)
            - quicksum(cf[(arco, k)]*y_s[arco[0], arco[1], k] for arco in AF for k in K)
            - quicksum(cv[(arco, k)]*size[s][item]*w_s[arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
            - quicksum(ch[arco]*size[s][item]*x_s[arco[0], arco[1], item] for item in Cargo for arco in AG) 
            )
    TRC_s = quicksum( sc[s][(arco, k)]*zplus_s[arco[0], arco[1], k] for k in K for arco in AF)

    RC_s = quicksum(tv*gap[arco, arco2]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k]
                    - sc[s][arco2, k]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

    obj = SSP_s - TRC_s - RC_s
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()

    return ss6, ss7, ss10, ss11



################
# SOLVE MODELS #
################


def solve_integer_Qs_SingleSat(y0_param, s, A, K, int_r1, int_sat, partial_gap=None):
    #update RHS:
    for i,j in A:
        for k in K:
            int_r1[i, j, k].RHS = round(y0_param[i, j, k])
    if partial_gap != None:
        int_sat.setParam('MIPGap', partial_gap)
    #solve model
    int_sat.optimize()

    status = int_sat.status
    if status == GRB.Status.INFEASIBLE:
        print('Satellite model was stopped with status {}'.format(status))
        # compute Irreducible Inconsistent Subsystem
        int_sat.computeIIS()
        for constr in int_sat.getConstrs():
            if constr.IISConstr:
                print('Infeasible constraint: {}'.format(constr.constrName))

    if partial_gap != None:
        return int_sat.ObjVal, int_sat.ObjBound, int_sat.MIPGap
    else:
        return int_sat.ObjVal, int_sat.ObjBound





def solve_relaxed_Qs_SingleSat(y0_param, s, A, AF, K, rel_r1, rel_sat):
    #update RHS:
    for i,j in A:
        for k in K:
            rel_r1[i, j, k].RHS = round(y0_param[i, j, k])
    #solve model
    rel_sat.optimize()

    status = rel_sat.status
    if status == GRB.Status.INFEASIBLE:
        print('Satellite model was stopped with status {}'.format(status))
        # compute Irreducible Inconsistent Subsystem
        rel_sat.computeIIS()
        for constr in rel_sat.getConstrs():
            if constr.IISConstr:
                print('Infeasible constraint: {}'.format(constr.constrName))

    #get dual variables
    PI_r1 = {}
    for i,j in AF:
        for k in K:
            PI_r1[i, j, k] = rel_r1[i, j, k].Pi

    return rel_sat.ObjVal, PI_r1





def solve_integer_Qs(y0_param, s, A, K, int_r1, int_sat, partial_gap=None):
    #update RHS:
    for i,j in A:
        for k in K:
            int_r1[s][i, j, k].RHS = round(y0_param[i, j, k])
    if partial_gap != None:
        int_sat[s].setParam('MIPGap', partial_gap)
    #solve model
    int_sat[s].optimize()

    status = int_sat[s].status
    if status == GRB.Status.INFEASIBLE:
        print('Satellite model was stopped with status {}'.format(status))
        # compute Irreducible Inconsistent Subsystem
        int_sat[s].computeIIS()
        for constr in int_sat[s].getConstrs():
            if constr.IISConstr:
                print('Infeasible constraint: {}'.format(constr.constrName))

    if partial_gap != None:
        return int_sat[s].ObjVal, int_sat[s].ObjBound, int_sat[s].MIPGap
    else:
        return int_sat[s].ObjVal, int_sat[s].ObjBound





def solve_relaxed_Qs(y0_param, s, A, AF, K, rel_r1, rel_sat):
    #update RHS:
    for i,j in A:
        for k in K:
            rel_r1[s][i, j, k].RHS = round(y0_param[i, j, k])
    #solve model
    rel_sat[s].optimize()
    #get dual variables
    PI_r1 = {}
    for i,j in AF:
        for k in K:
            PI_r1[i, j, k] = rel_r1[s][i, j, k].Pi

    return rel_sat[s].ObjVal, PI_r1


#################
# CREATE MODELS #
#################


def create_second_stage_satellites_FSC_SingleSat(days, s, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, integer, threads=1, continuous_cargo=False, volperkg=None, incperkg=None):
    ########################
    ### Satellite Models ###
    ########################
    model_s = None  #modelSingleSat
    alpha_constraint_s = None  # constraint1: alpha_y = y_param in modelSingleSat

    if integer:
        model = Model("satellite SingleSat INTEGER")

        #################
        ### Variables ###
        #################
        y_s = model.addVars(A, K, vtype=GRB.INTEGER, name='y{}'.format(s))
        zplus_s = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus{}'.format(s))
        zminus_s = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus{}'.format(s))
        if not continuous_cargo:
            x_s = model.addVars(A, Cargo, vtype=GRB.BINARY, name='x{}'.format(s))
            w_s = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
            q_s = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        else:
            f_s = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, lb=0.0, name='f{}'.format(s))
        #retiming
        r_s = {}
        for arco in AF:
            r_s[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
        #new variable for Benders cut
        #alpha_y[i, j, k] = y[0][i, j, k]
        alpha_y = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='alpha_y')
    else:
        model = Model("satellite SingleSat RELAXED")

        #################
        ### Variables ###
        #################        
        y_s = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='y{}'.format(s))
        zplus_s = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zplus{}'.format(s))
        zminus_s = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zminus{}'.format(s))
        if not continuous_cargo:
            x_s = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='x{}'.format(s))
            w_s = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
            q_s = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        else:
            f_s = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, lb=0.0, name='f{}'.format(s))
        #retiming
        r_s = {}
        for arco in AF:
            r_s[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
        #new variable for Benders cut
        #alpha_y[i, j, k] = y[0][i, j, k]
        alpha_y = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='alpha_y')        
        
    ###################
    ### Constraints ###
    ###################
    #new variables assignation, update RHS on each iteration
    rs1 = model.addConstrs((alpha_y[i, j, k] 
            == 0 for i,j in A for k in K), name="alpha_y")
    ## SECOND STAGE CONSTRAINTS ##
    #SS1
    model.addConstrs((quicksum(y_s[i, j, k] for k in K) 
            <= 1 for i,j in AF ), name='(SS1)')
    #SS2        
    model.addConstrs((y_s[i, j, k] == alpha_y[i, j, k]
            + zplus_s[i, j, k] - zminus_s[i, j, k] for k in K for i,j in A ), name="(SS2)")
    #SS3
    model.addConstrs((quicksum(y_s[i, j, k] for i,j in A if i == ii) 
            == nav[ii, k] for k in K for ii in Nf ), name='(SS3)')        
    #SS4
    model.addConstrs((quicksum(y_s[i, j, k] for i,j in A if j == jj) - quicksum(y_s[i, j, k] for i,j in A if i == jj) 
            == 0 for k in K for jj in Nint ), name='(SS4)')
    #SS5
    if not continuous_cargo:
        for jj in N:
            for item in Cargo:
                od = OD[item]
                if od[1] == jj:
                    #SS5.1
                    model.addConstr((quicksum(x_s[i, j, item] for i,j in A if j == jj) - quicksum(x_s[i, j, item] for i,j in A if i == jj) 
                            == q_s[item] ), name='(SS5.1)')
                elif od[0] == jj:
                    #SS5.2
                    model.addConstr((quicksum(x_s[i, j, item] for i,j in A if j == jj) - quicksum(x_s[i, j, item] for i,j in A if i == jj) 
                            == -q_s[item] ), name='(SS5.2)')
                else:
                    #SS5.3
                    model.addConstr((quicksum(x_s[i, j, item] for i,j in A if j == jj) - quicksum(x_s[i, j, item] for i,j in A if i == jj) 
                            == 0 ), name='(SS5.3)')
    else:
        ss5_list = []
        for jj in N:
            for item in Cargo:
                od = OD[item]
                if mand[0][item]:
                    if od[1] == jj:
                        #SS5.1
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            == size[s][item]), name='(SS5.1)')
                    elif od[0] == jj:
                        #SS5.2
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            == -size[s][item]), name='(SS5.2)')
                    else:
                        #SS5.3
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            == 0), name='(SS5.3)')
                else:
                    if od[1] == jj:
                        #SS5.1
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            <= size[s][item]), name='(SS5.1)')
                    elif od[0] == jj:
                        #SS5.2
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            >= -size[s][item]), name='(SS5.2)')
                    else:
                        #SS5.3
                        ss5 = model.addConstr((quicksum(f_s[i, j, item] for i,j in A if j == jj) - quicksum(f_s[i, j, item] for i,j in A if i == jj) 
                            == 0), name='(SS5.3)')
                ss5_list.append(ss5)
    #SS6
    if not continuous_cargo:
        ss6 = model.addConstrs((quicksum(size[s][item]*x_s[i, j, item] for item in Cargo)
                <= quicksum(air_cap[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS6)")
    else:
        model.addConstrs((quicksum(f_s[i, j, item] for item in Cargo)
                <= quicksum(air_cap[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS6)")
    #SS7
    if not continuous_cargo:
        ss7 = model.addConstrs((quicksum(vol[s][item]*x_s[i, j, item] for item in Cargo)
                <= quicksum(air_vol[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS7)")
    else:
        ss7 = model.addConstrs((quicksum(volperkg[s][item]*f_s[i, j, item] for item in Cargo)
                <= quicksum(air_vol[k]*y_s[i, j, k] for k in K) for i,j in AF ), name="(SS7)")
    #SS8
    if not continuous_cargo:
        model.addConstrs((x_s[i, j, item] 
                <= quicksum(y_s[i, j, k] for k in K) for item in Cargo for i,j in AF ), name="(SS8)")
    else:
        model.addConstrs((f_s[i, j, item]
                <= quicksum(air_cap[k]*y_s[i, j, k] for k in K) for item in Cargo for i,j in AF ), name="(SS8)")

    if not continuous_cargo:
        #SS9 
        model.addConstrs((x_s[i, j, item] + y_s[i, j, k] 
                <= w_s[i, j, item, k] + 1 for k in K for item in Cargo for i,j in AF ), name="(SS9)")        
        #SS10
        ss10 = model.addConstrs((q_s[item] 
                >= mand[s][item] for item in Cargo ), name="(SS10)")
        #SS11
        ss11 = model.addConstrs((q_s[item] 
                <= ex[s][item] for item in Cargo ), name="(SS11)")
    #SS12
    model.addConstrs((zminus_s[i, j, k] 
            <= alpha_y[i, j, k] for k in K for i,j in AF), name="(SS11)")  
    #SS13
    model.addConstrs((alpha_y[i, j, k] + zplus_s[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(SS12)")                 
    #z+_bin_flight
    model.addConstrs((zplus_s[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z+_bin_flight)")
    #z-_bin_flight
    model.addConstrs((zminus_s[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z-_bin_flight)")        
    # Retiming       
    #RT1
    model.addConstrs((alpha_y[i, j, k] + zminus_s[i, j, k] + zplus_s[i2, j2, k]
            <= 2 + r_s[i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j] ), name="(RT1)")		        
    #RT2
    model.addConstrs((quicksum(r_s[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= alpha_y[i, j, k] for k in K for i,j in AF), name="(RT2)")		        
    #RT3
    model.addConstrs((quicksum(r_s[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= zminus_s[i, j, k] for k in K for i,j in AF), name="(RT3)")      
    #RT4
    model.addConstrs((quicksum(r_s[i, j][i2, j2, k] for i,j in V[i2, j2])
            <= zplus_s[i2, j2, k] for k in K for i2,j2 in AF), name="(RT4)") 

    ##########################
    ### Objective Function ###
    ##########################
    # Definition of the second stage value        
    #SSP second stage profit for each scenario
    #TRC total recovery cost
    #RC re-timing costs
    if not continuous_cargo:
        SSP_s = (  quicksum(inc[s][item]*q_s[item] for item in Cargo)
                - quicksum(cf[(arco, k)]*y_s[arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, k)]*size[s][item]*w_s[arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                - quicksum(ch[arco]*size[s][item]*x_s[arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    else:
        SSP_s = (  quicksum(incperkg[s][item]*quicksum(f_s[i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
                - quicksum(cf[(arco, k)]*y_s[arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, 1)]*f_s[arco[0], arco[1], item] for item in Cargo for arco in AF)
                - quicksum(ch[arco]*f_s[arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    TRC_s = quicksum( sc[s][(arco, k)]*zplus_s[arco[0], arco[1], k] for k in K for arco in AF)

    RC_s = quicksum(tv*gap[arco, arco2]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k]
                    - sc[s][arco2, k]*r_s[arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

    obj = SSP_s - TRC_s - RC_s
    model.setObjective(obj, GRB.MAXIMIZE)
    model.setParam('OutputFlag', False)
    if threads != None:
        model.Params.Threads = threads
    # model.setParam("LogFile", "log {} {} threads.log".format(model.ModelName, model.Params.Threads))
    if sat_gap != None:
        model.setParam('MIPGap', sat_gap)
    model.update()

    alpha_constraint_s = rs1
    model_s = model

    if not continuous_cargo:
        return model_s, alpha_constraint_s, y_s, x_s, w_s, q_s, zplus_s, r_s, ss6, ss7, ss10, ss11
    else:
        return model_s, alpha_constraint_s, y_s, f_s, zplus_s, r_s, ss5_list, ss7





def create_second_stage_satellites_FSC(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, integer, threads=1, continuous_cargo=False, volperkg=None, incperkg=None):
    ########################
    ### Satellite Models ###
    ########################
    models = {}  #{s: model_s}
    alpha_constraints = {}  #{s: constraint1: alpha_y = y_param in model_s}

    for s in S:
        if integer:
            model = Model("satellite d{}s{} scenario {} INTEGER".format(days, len(S), s))

            #################
            ### Variables ###
            #################
            y = {}  #{s: y[i, j, k]}
            # second stage
            if not continuous_cargo:
                x = {}  #{s: {x[i, j, od]}}
                q = {}  #{s: {q[od]}
                w = {}  #{s: {w[i, j, od, k]}}   
                x[s] = model.addVars(A, Cargo, vtype=GRB.BINARY, name='x{}'.format(s))
                w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
                q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
            else:
                f = {}  #{s: {f[i, j, od]}}
                f[s] = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, lb=0.0, name='f{}'.format(s))
            zplus = {}  #{s: {k: zplus[arco, m]}}
            zminus = {}  #{s: {k: zminus[arco, m]}}
            y[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='y{}'.format(s))
            zplus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus{}'.format(s))
            zminus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus{}'.format(s))
            #retiming
            r = {}  # {s: {arco: r[arco2, k]}},  r[s][arco][V[arco], k]
            rs = {}
            for arco in AF:
                rs[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
            r[s] = rs
            #new variable for Benders cut
            #alpha_y[i, j, k] = y[0][i, j, k]
            alpha_y = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='alpha_y')
        else:
            model = Model("satellite d{}s{} scenario {} RELAXED".format(days, len(S), s))

            #################
            ### Variables ###
            #################        
            y = {}  #{s: y[i, j, k]}
            # second stage
            if not continuous_cargo:
                x = {}  #{s: {x[i, j, od]}}
                q = {}  #{s: {q[od]}
                w = {}  #{s: {w[i, j, od, k]}}   
                x[s] = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='x{}'.format(s))
                w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
                q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
            else:
                f = {}  #{s: {f[i, j, od]}}
                f[s] = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, lb=0.0, name='f{}'.format(s))
            zplus = {}  #{s: {k: zplus[arco, m]}}
            zminus = {}  #{s: {k: zminus[arco, m]}}
            y[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='y{}'.format(s))
            zplus[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zplus{}'.format(s))
            zminus[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zminus{}'.format(s))
            #retiming
            r = {}  # {s: {arco: r[arco2, k]}},  r[s][arco][V[arco], k]
            rs = {}
            for arco in AF:
                rs[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
            r[s] = rs
            #new variable for Benders cut
            #alpha_y[i, j, k] = y[0][i, j, k]
            alpha_y = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='alpha_y')        
            
        ###################
        ### Constraints ###
        ###################
        #new variables assignation, update RHS on each iteration
        rs1 = model.addConstrs((alpha_y[i, j, k] 
                == 0 for i,j in A for k in K), name="alpha_y")
        ## SECOND STAGE CONSTRAINTS ##
        #SS1
        model.addConstrs((quicksum(y[s][i, j, k] for k in K) 
                <= 1 for i,j in AF ), name='(SS1)')
        #SS2        
        model.addConstrs((y[s][i, j, k] == alpha_y[i, j, k]
                + zplus[s][i, j, k] - zminus[s][i, j, k] for k in K for i,j in A ), name="(SS2)")
        #SS3
        model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if i == ii) 
                == nav[ii, k] for k in K for ii in Nf ), name='(SS3)')        
        #SS4
        model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if j == jj) - quicksum(y[s][i, j, k] for i,j in A if i == jj) 
                == 0 for k in K for jj in Nint ), name='(SS4)')
        #SS5
        if not continuous_cargo:
            for jj in N:
                for item in Cargo:
                    od = OD[item]
                    if od[1] == jj:
                        #SS5.1
                        model.addConstr((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                                == q[s][item] ), name='(SS5.1)')
                    elif od[0] == jj:
                        #SS5.2
                        model.addConstr((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                                == -q[s][item] ), name='(SS5.2)')
                    else:
                        #SS5.3
                        model.addConstr((quicksum(x[s][i, j, item] for i,j in A if j == jj) - quicksum(x[s][i, j, item] for i,j in A if i == jj) 
                                == 0 ), name='(SS5.3)')
        else:
            for jj in N:
                for item in Cargo:
                    od = OD[item]
                    if mand[0][item] == 1:
                        if od[1] == jj:
                            #SS5.1
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                == size[s][item] ), name='(SS5.1)')
                        elif od[0] == jj:
                            #SS5.2
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                == -size[s][item] ), name='(SS5.2)')
                        else:
                            #SS5.3
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                == 0 ), name='(SS5.3)')
                    else:
                        if od[1] == jj:
                            #SS5.1
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                <= size[s][item] ), name='(SS5.1)')
                        elif od[0] == jj:
                            #SS5.2
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                >= -size[s][item] ), name='(SS5.2)')
                        else:
                            #SS5.3
                            model.addConstr((quicksum(f[s][i, j, item] for i,j in A if j == jj) - quicksum(f[s][i, j, item] for i,j in A if i == jj) 
                                == 0 ), name='(SS5.3)')
        #SS6
        if not continuous_cargo:
            model.addConstrs((quicksum(size[s][item]*x[s][i, j, item] for item in Cargo)
                    <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS6)")
        else:
            model.addConstrs((quicksum(f[s][i, j, item] for item in Cargo)
                    <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS6)")
        #SS7
        if not continuous_cargo:
            model.addConstrs((quicksum(vol[s][item]*x[s][i, j, item] for item in Cargo)
                    <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS7)")
        else:
            model.addConstrs((quicksum(volperkg[s][item]*f[s][i, j, item] for item in Cargo)
                    <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS7)")
        #SS8
        if not continuous_cargo:
            model.addConstrs((x[s][i, j, item] 
                    <= quicksum(y[s][i, j, k] for k in K) for item in Cargo for i,j in AF ), name="(SS8)")
        else:
            model.addConstrs((f[s][i, j, item]
                    <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for item in Cargo for i,j in AF ), name="(SS8)")

        if not continuous_cargo:            
            #SS9 
            model.addConstrs((x[s][i, j, item] + y[s][i, j, k] 
                    <= w[s][i, j, item, k] + 1 for k in K for item in Cargo for i,j in AF ), name="(SS9)")        
            #SS10
            model.addConstrs((q[s][item] 
                    >= mand[s][item] for item in Cargo ), name="(SS10)")
            #SS11
            model.addConstrs((q[s][item] 
                    <= ex[s][item] for item in Cargo ), name="(SS11)")
        #SS12
        model.addConstrs((zminus[s][i, j, k] 
                <= alpha_y[i, j, k] for k in K for i,j in AF), name="(SS12)")  
        #SS13
        model.addConstrs((alpha_y[i, j, k] + zplus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(SS13)")                 
        #z+_bin_flight
        model.addConstrs((zplus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(z+_bin_flight)")
        #z-_bin_flight
        model.addConstrs((zminus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(z-_bin_flight)")        
        # Retiming       
        #RT1
        model.addConstrs((alpha_y[i, j, k] + zminus[s][i, j, k] + zplus[s][i2, j2, k]
                <= 2 + r[s][i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j] ), name="(RT1)")		        
        #RT2
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
                <= alpha_y[i, j, k] for k in K for i,j in AF), name="(RT2)")		        
        #RT3
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
                <= zminus[s][i, j, k] for k in K for i,j in AF), name="(RT3)")      
        #RT4
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i,j in V[i2, j2])
                <= zplus[s][i2, j2, k] for k in K for i2,j2 in AF), name="(RT4)") 

        ##########################
        ### Objective Function ###
        ##########################
        # Definition of the second stage value        
        SSP = {}  #{s: SSPs} second stage profit for each scenario
        TRC = {}  #{s: TRCs} total recovery cost
        RC = {}  #{s: RCs} re-timing costs
        if not continuous_cargo:
            SSP[s] = (  quicksum(inc[s][item]*q[s][item] for item in Cargo)
                    - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                    - quicksum(cv[(arco, k)]*size[s][item]*w[s][arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                    - quicksum(ch[arco]*size[s][item]*x[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                    )
        else:
            SSP[s] = (  quicksum(incperkg[s][item]*quicksum(f[s][i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
                    - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                    - quicksum(cv[(arco, 1)]*f[s][arco[0], arco[1], item] for item in Cargo for arco in AF)
                    - quicksum(ch[arco]*f[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                    )
        TRC[s] = quicksum( sc[s][(arco, k)]*zplus[s][arco[0], arco[1], k] for k in K for arco in AF)

        RC[s] = quicksum(tv*gap[arco, arco2]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k]
                        - sc[s][arco2, k]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

        obj = SSP[s] - TRC[s] - RC[s]
        model.setObjective(obj, GRB.MAXIMIZE)
        model.setParam('OutputFlag', False)
        if threads != None:
            model.Params.Threads = threads
        # model.setParam("LogFile", "log {} {} threads.log".format(model.ModelName, model.Params.Threads))
        if sat_gap != None:
            model.setParam('MIPGap', sat_gap)
        model.update()

        alpha_constraints[s] = rs1
        models[s] = model

    return models, alpha_constraints




if __name__ == "__main__":
    print("Modulo con funciones para crear modelos de segunda etapa FSC")