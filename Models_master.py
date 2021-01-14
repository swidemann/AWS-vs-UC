#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

from gurobipy import Model, GRB, quicksum

master_gap = 0.01
MIPFocus = None  # 1: feasible solutions, 2: optimality, 3: bound
NodefileStart = 0.5  # write nodes to disk after x GB, recomended 0.5


def create_master_FSC(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, lc, model_name, instance, L, L_method, master_time, lazy=True):
    
    ####################
    ### Master Model ###
    ####################

    model = Model(model_name)
    log_name = "log {}.log".format(model.ModelName)
    model.setParam("LogFile", log_name)

    #To add feasible solutions after solving integer Q(y)
    model._sol_to_add = []  # list y0
    model._theta_to_add = 0  # float theta
    model._current_incumbent = 0 # float -FSC + Q_int

    #track the cuts actually aded (not all the computed are added)
    model._Q_cblazy_added = 0
    model._Q_relaxed_cblazy_added = 0


    #################
    ### Variables ###
    #################

    y = {}  #{s: y[i, j, k]}
    # first stage
    y0 = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0'))
    y[0] = y0
    theta = model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name='theta')

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
    #Upper bound for theta
    model.addConstr((theta
            <= L), name="theta<=L")                    

    ##########################
    ### Objective Function ###
    ##########################

    # Definition of the first stage costs
    FSC = quicksum(lc[(arco, k)]*y[0][arco[0], arco[1], k] for arco in AF for k in K) 

    obj = -FSC + theta
    model.setObjective(obj, GRB.MAXIMIZE)
    model.Params.Threads = 1
    if lazy:
        model.Params.lazyConstraints = 1
    if master_gap != None:
        model.setParam('MIPGap', master_gap)
    if master_time != None:
        model.setParam('TimeLimit', master_time)
    if MIPFocus != None:
        model.setParam('MIPFocus', MIPFocus)
    if NodefileStart != None:
        model.setParam('NodefileStart', NodefileStart)
    model.update()

    return model, log_name, y, y0, theta, FSC





def create_master_FSC_sb(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, model_name, instance, L, L_method, master_time, volperkg_sb, incperkg_sb, lazy=True):

    ####################
    ### Master Model ###
    ####################

    model = Model(model_name)
    log_name = "log {}.log".format(model.ModelName)
    model.setParam("LogFile", log_name)

    #To add feasible solutions after solving integer Q(y)
    model._sol_to_add = []  # list y0
    model._theta_to_add = 0  # float theta
    model._current_incumbent = 0 # float -FSC + Q_int

    #track the cuts actually aded (not all the computed are added)
    model._Q_cblazy_added = 0
    model._Q_relaxed_cblazy_added = 0
    model._QTbound_cblazy_added = 0


    #################
    ### Variables ###
    #################

    y = {}  #{s: y[i, j, k]}
    # first stage
    y0 = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0'))
    y[0] = y0
    theta = model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name='theta')
    # bound theta with a real second stage, min{vol, cv} and max{inc, density}, but selecting one scenario for size
    y_bound = model.addVars(A, K, vtype=GRB.INTEGER, name=('y_bound'))
    f_bound = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, name='f_bound')
    g_bound = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, name='g_bound')
    zplus_bound = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus_bound')
    zminus_bound = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus_bound')
    sel_size = model.addVars(Cargo, vtype=GRB.CONTINUOUS, name='sel_size')
    #retiming
    r_bound = {}
    for arco in AF:
        r_bound[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{})_bound'.format(arco[0], arco[1]))
    #select only one scenario to bound
    gamma = model.addVars(S, vtype=GRB.INTEGER, name='gamma')

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
    #Upper bound for theta
    model.addConstr((theta
            <= L), name="theta<=L")  
    ## BOUND BY SELECTING AND SOLVING ONE SCENARIO ##
    #B1
    model.addConstrs((quicksum(y_bound[i, j, k] for k in K) 
            <= 1 for i,j in AF), name='(B1)')
    #B2        
    model.addConstrs((y_bound[i, j, k] == y[0][i, j, k]
            + zplus_bound[i, j, k] - zminus_bound[i, j, k] for k in K for i,j in A), name="(B2)")
    #B3
    model.addConstrs((quicksum(y_bound[i, j, k] for i,j in A if i == ii) 
            == nav[ii, k] for k in K for ii in Nf), name='(B3)')        
    #B4
    model.addConstrs((quicksum(y_bound[i, j, k] for i,j in A if j == jj) - quicksum(y_bound[i, j, k] for i,j in A if i == jj) 
            == 0 for k in K for jj in Nint), name='(B4)')
    #B5
    for jj in N:
        for item in Cargo:
            od = OD[item]
            if mand[0][item]:
                if od[1] == jj:
                    #B5.1
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == sel_size[item]), name='(B5.1)')
                elif od[0] == jj:
                    #B5.2
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == -sel_size[item]), name='(B5.2)')
                else:
                    #B5.3
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(B5.3)')
            else:
                if od[1] == jj:
                    #B5.1
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        <= sel_size[item]), name='(B5.1)')
                elif od[0] == jj:
                    #B5.2
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        >= -sel_size[item]), name='(B5.2)')
                else:
                    #B5.3
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(B5.3)')
    #B6
    model.addConstrs((quicksum(f_bound[i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y_bound[i, j, k] for k in K) for i,j in AF), name="(B6)")
    #B7
    model.addConstrs((quicksum(g_bound[i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y_bound[i, j, k] for k in K) for i,j in AF), name="(B7)")
    #B8
    model.addConstrs((g_bound[i, j, item]
            == volperkg_sb[item]*f_bound[i, j, item] for item in Cargo for i,j in A), name="(B8)")  
    #B9
    model.addConstrs((sel_size[item]
            == quicksum(size[s][item]*gamma[s] for s in S) for item in Cargo), name="(B9)")                
    #B12
    model.addConstrs((zminus_bound[i, j, k] 
            <= y[0][i, j, k] for k in K for i,j in AF), name="(B12)")  
    #B13
    model.addConstrs((y[0][i, j, k] + zplus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(B13)")                 
    #B19 z+_bin_flight
    model.addConstrs((zplus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z+_bin_flight)")
    #B21 z-_bin_flight
    model.addConstrs((zminus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z-_bin_flight)")        
    # Retiming       
    #RT1
    model.addConstrs((y[0][i, j, k] + zminus_bound[i, j, k] + zplus_bound[i2, j2, k]
            <= 2 + r_bound[i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j]), name="(RT1)")		        
    #RT2
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= y[0][i, j, k] for k in K for i,j in AF), name="(RT2)")		        
    #RT3
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= zminus_bound[i, j, k] for k in K for i,j in AF), name="(RT3)")      
    #RT4
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i,j in V[i2, j2])
            <= zplus_bound[i2, j2, k] for k in K for i2,j2 in AF), name="(RT4)")
    # Definition of the second stage value        
    SSP_bound = (  quicksum(incperkg_sb[item]*quicksum(f_bound[i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
                - quicksum(cf[arco, k]*y_bound[arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum( min(cv[(arco, k)] for k in K)*f_bound[arco[0], arco[1], item] for item in Cargo for arco in AF)
                - quicksum(ch[arco]*f_bound[arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    TRC_bound = quicksum( min(sc[s][(arco, k)] for s in S)*zplus_bound[arco[0], arco[1], k] for k in K for arco in AF)

    RC_bound = quicksum(tv*gap[arco, arco2]*r_bound[arco[0], arco[1]][arco2[0], arco2[1], k]
                    - min(sc[s][(arco, k)] for s in S)*r_bound[arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])
    #BQ
    Q_bound = SSP_bound - TRC_bound - RC_bound
    #Upper bound for theta with average scenario
    model.addConstr((theta
            <= Q_bound), name="theta<=Q_bound")   
    #B10 Only one gamma
    model.addConstr((quicksum(gamma[s] for s in S)
            == 1), name="(One_gamma)")                           

    ##########################
    ### Objective Function ###
    ##########################

    # Definition of the first stage costs
    FSC = quicksum(lc[(arco, k)]*y[0][arco[0], arco[1], k] for arco in AF for k in K) 

    obj = -FSC + theta
    model.setObjective(obj, GRB.MAXIMIZE)
    model.Params.Threads = 1
    if lazy:
        model.Params.lazyConstraints = 1
    if master_gap != None:
        model.setParam('MIPGap', master_gap)
    if master_time != None:
        model.setParam('TimeLimit', master_time)
    if MIPFocus != None:
        model.setParam('MIPFocus', MIPFocus)
    if NodefileStart != None:
        model.setParam('NodefileStart', NodefileStart)
    model.update()

    return model, log_name, y, y0, theta, FSC





def create_master_FSC_sb_ContFlow(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, model_name, instance, L, L_method, master_time, volperkg_sb, incperkg_sb, lazy=True):

    ####################
    ### Master Model ###
    ####################

    model = Model(model_name)
    log_name = "log {}.log".format(model.ModelName)
    model.setParam("LogFile", log_name)

    #To add feasible solutions after solving integer Q(y)
    model._sol_to_add = []  # list y0
    model._theta_to_add = 0  # float theta
    model._current_incumbent = 0 # float -FSC + Q_int

    #track the cuts actually aded (not all the computed are added)
    model._Q_cblazy_added = 0
    model._Q_relaxed_cblazy_added = 0
    model._QTbound_cblazy_added = 0


    #################
    ### Variables ###
    #################

    y = {}  #{s: y[i, j, k]}
    # first stage
    y0 = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0'))
    y[0] = y0
    theta = model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name='theta')
    # bound theta with a real second stage, min{vol, cv} and max{inc, density}, but selecting one scenario for size
    y_bound = model.addVars(A, K, vtype=GRB.INTEGER, name=('y_bound'))
    f_bound = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, name='f_bound')
    zplus_bound = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus_bound')
    zminus_bound = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus_bound')
    sel_size = model.addVars(Cargo, vtype=GRB.CONTINUOUS, name='sel_size')
    #retiming
    r_bound = {}
    for arco in AF:
        r_bound[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{})_bound'.format(arco[0], arco[1]))
    #select only one scenario to bound
    gamma = model.addVars(S, vtype=GRB.INTEGER, name='gamma')

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
    #Upper bound for theta
    model.addConstr((theta
            <= L), name="theta<=L")  
    ## BOUND BY SELECTING AND SOLVING ONE SCENARIO ##
    #B1
    model.addConstrs((quicksum(y_bound[i, j, k] for k in K) 
            <= 1 for i,j in AF), name='(B1)')
    #B2        
    model.addConstrs((y_bound[i, j, k] == y[0][i, j, k]
            + zplus_bound[i, j, k] - zminus_bound[i, j, k] for k in K for i,j in A), name="(B2)")
    #B3
    model.addConstrs((quicksum(y_bound[i, j, k] for i,j in A if i == ii) 
            == nav[ii, k] for k in K for ii in Nf), name='(B3)')        
    #B4
    model.addConstrs((quicksum(y_bound[i, j, k] for i,j in A if j == jj) - quicksum(y_bound[i, j, k] for i,j in A if i == jj) 
            == 0 for k in K for jj in Nint), name='(B4)')
    #B5
    for jj in N:
        for item in Cargo:
            od = OD[item]
            if mand[0][item]:
                if od[1] == jj:
                    #B5.1
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == sel_size[item]), name='(B5.1)')
                elif od[0] == jj:
                    #B5.2
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == -sel_size[item]), name='(B5.2)')
                else:
                    #B5.3
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(B5.3)')
            else:
                if od[1] == jj:
                    #B5.1
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        <= sel_size[item]), name='(B5.1)')
                elif od[0] == jj:
                    #B5.2
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        >= -sel_size[item]), name='(B5.2)')
                else:
                    #B5.3
                    model.addConstr((quicksum(f_bound[i, j, item] for i,j in A if j == jj) - quicksum(f_bound[i, j, item] for i,j in A if i == jj) 
                        == 0), name='(B5.3)')
    #B6
    model.addConstrs((quicksum(f_bound[i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y_bound[i, j, k] for k in K) for i,j in AF), name="(B6)")
    #B7
    model.addConstrs((quicksum(f_bound[i, j, item]*volperkg_sb[item] for item in Cargo)
            <= quicksum(air_vol[k]*y_bound[i, j, k] for k in K) for i,j in AF), name="(B7)")
    #B8
    model.addConstrs((f_bound[i, j, item]
            <= quicksum(air_cap[k]*y_bound[i, j, k] for k in K) for item in Cargo for i,j in AF), name="(B8)")
    #B9
    model.addConstrs((sel_size[item]
            == quicksum(size[s][item]*gamma[s] for s in S) for item in Cargo), name="(B9)")                
    #B12
    model.addConstrs((zminus_bound[i, j, k] 
            <= y[0][i, j, k] for k in K for i,j in AF), name="(B12)")  
    #B13
    model.addConstrs((y[0][i, j, k] + zplus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(B13)")                 
    #B19 z+_bin_flight
    model.addConstrs((zplus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z+_bin_flight)")
    #B21 z-_bin_flight
    model.addConstrs((zminus_bound[i, j, k] 
            <= 1 for k in K for i,j in AF), name="(z-_bin_flight)")        
    # Retiming       
    #RT1
    model.addConstrs((y[0][i, j, k] + zminus_bound[i, j, k] + zplus_bound[i2, j2, k]
            <= 2 + r_bound[i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j]), name="(RT1)")		        
    #RT2
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= y[0][i, j, k] for k in K for i,j in AF), name="(RT2)")		        
    #RT3
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= zminus_bound[i, j, k] for k in K for i,j in AF), name="(RT3)")      
    #RT4
    model.addConstrs((quicksum(r_bound[i, j][i2, j2, k] for i,j in V[i2, j2])
            <= zplus_bound[i2, j2, k] for k in K for i2,j2 in AF), name="(RT4)")
    # Definition of the second stage value        
    SSP_bound = (  quicksum(incperkg_sb[item]*quicksum(f_bound[i, j, item] for i,j in A if j == OD[item][1]) for item in Cargo)
                - quicksum(cf[arco, k]*y_bound[arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum( min(cv[(arco, k)] for k in K)*f_bound[arco[0], arco[1], item] for item in Cargo for arco in AF)
                - quicksum(ch[arco]*f_bound[arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
    TRC_bound = quicksum( min(sc[s][(arco, k)] for s in S)*zplus_bound[arco[0], arco[1], k] for k in K for arco in AF)

    RC_bound = quicksum(tv*gap[arco, arco2]*r_bound[arco[0], arco[1]][arco2[0], arco2[1], k]
                    - min(sc[s][(arco, k)] for s in S)*r_bound[arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])
    #BQ
    Q_bound = SSP_bound - TRC_bound - RC_bound
    #Upper bound for theta with average scenario
    model.addConstr((theta
            <= Q_bound), name="theta<=Q_bound")   
    #B10 Only one gamma
    model.addConstr((quicksum(gamma[s] for s in S)
            == 1), name="(One_gamma)")                           

    ##########################
    ### Objective Function ###
    ##########################

    # Definition of the first stage costs
    FSC = quicksum(lc[(arco, k)]*y[0][arco[0], arco[1], k] for arco in AF for k in K) 

    obj = -FSC + theta
    model.setObjective(obj, GRB.MAXIMIZE)
    model.Params.Threads = 1
    if lazy:
        model.Params.lazyConstraints = 1
    if master_gap != None:
        model.setParam('MIPGap', master_gap)
    if master_time != None:
        model.setParam('TimeLimit', master_time)
    if MIPFocus != None:
        model.setParam('MIPFocus', MIPFocus)
    if NodefileStart != None:
        model.setParam('NodefileStart', NodefileStart)
    model.update()

    return model, log_name, y, y0, theta, FSC
