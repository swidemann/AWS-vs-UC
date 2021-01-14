#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019
Solve the relaxed DEF with only second stage income and cost, 
and without first stage profit nor second stage recovery/retiming/offloading costs

@author: steffan
"""

from gurobipy import Model, GRB, quicksum
import time


L_time = 5*60  # time limit to compute L
L_gap = 0.01


def compute_L_SSLP(instance, days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, write_log=False):

    start_time = time.time()
    #############
    ### Model ###
    #############
    model = Model()
    if not write_log:
        model.setParam('OutputFlag', False)
    else:
        model.setParam("LogFile", "log L SSLP {}.log".format(instance))

    #################
    ### Variables ###
    #################
    # first stage
    y = {}  #{s: y[i, j, k]}
    y0 = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0'))
    y[0] = y0
    # second stage
    x = {}  #{s: {x[i, j, od]}}
    q = {}  #{s: {q[od]}
    w = {}  #{s: {w[i, j, od, k]}}
    zplus = {}  #{s: {k: zplus[arco, m]}}
    zminus = {}  #{s: {k: zminus[arco, m]}}
    for s in S:
        y[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name=('y{}'.format(s)))
        zplus[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zplus{}'.format(s))
        zminus[s] = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='zminus{}'.format(s))
        x[s] = model.addVars(A, Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='x{}'.format(s))
        q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
    #retiming
    r = {}  # {s: {arco: r[arco2, k]}},  r[s][arco][V[arco], k]
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
    #SS6
    model.addConstrs((quicksum(size[s][item]*x[s][i, j, item] for item in Cargo)
            <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS6)")
    #SS7
    model.addConstrs((quicksum(vol[s][item]*x[s][i, j, item] for item in Cargo)
            <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS7)")
    #SS8
    model.addConstrs((x[s][i, j, item] 
            <= quicksum(y[s][i, j, k] for k in K) for item in Cargo for i,j in AF for s in S), name="(SS8)")
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
        SSP[s] = (  quicksum(inc[s][item]*q[s][item] for item in Cargo)
                - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, k)]*size[s][item]*w[s][arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                - quicksum(ch[arco]*size[s][item]*x[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
        TRC[s] = quicksum( sc[s][(arco, k)]*zplus[s][arco[0], arco[1], k] for k in K for arco in AF)

        RC[s] = quicksum(tv*gap[arco, arco2]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k]
                        - sc[s][arco2, k]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])
    Q = quicksum(SSP[s] - TRC[s] - RC[s] for s in S)/len(S)            
    obj = -FSC + Q
    model.setObjective(obj, GRB.MAXIMIZE)
    model.Params.Threads = 1
    if L_time != None:
        model.setParam('TimeLimit', L_time)
    if L_gap != None:
        model.setParam('MIPGap', L_gap)        
    model.update()
    model.optimize()
    run_time = time.time() - start_time
    print("****L = {} computed in {}s****".format(model.ObjBound, run_time))

    return model.objVal, model.ObjBound, model.status





def compute_L_anticipativity_integer_no_capacity(instance, days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, write_log=False):
    """
    Integer model without non-anticipativity constraints, and without capacity constrains
    (each scenario has its own first stage solution)
    """

    start_time = time.time()
    #############
    ### Model ###
    #############
    model = Model()
    if not write_log:
        model.setParam('OutputFlag', False)   
    else:
        model.setParam("LogFile", "log L Integer anticipativity {}.log".format(instance))        
    #################
    ### Variables ###
    #################
    # first stage
    y0 = {}
    for s in S:
        y0[s] = model.addVars(A, K, vtype=GRB.INTEGER, name=('y0{}'.format(s)))
    # second stage
    y = {}  #{s: y[i, j, k]}    
    x = {}  #{s: {x[i, j, od]}}
    q = {}  #{s: {q[od]}
    w = {}  #{s: {w[i, j, od, k]}}
    zplus = {}  #{s: {k: zplus[arco, m]}}
    zminus = {}  #{s: {k: zminus[arco, m]}}
    for s in S:
        y[s] = model.addVars(A, K, vtype=GRB.INTEGER, name=('y{}'.format(s)))
        zplus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus{}'.format(s))
        zminus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus{}'.format(s))
        x[s] = model.addVars(A, Cargo, vtype=GRB.BINARY, name='x{}'.format(s))
        q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
    #retiming
    r = {}  # {s: {arco: r[arco2, k]}},  r[s][arco][V[arco], k]
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
    model.addConstrs((quicksum(y0[s][i, j, k] for k in K) 
            <= 1 for i, j in AF for s in S), name='(FS1)')
    #FS2                
    model.addConstrs((quicksum(y0[s][i, j, k] for i,j in A if i == ii) 
            == nav[ii, k] for k in K for ii in Nf for s in S), name='(FS2)')
    #FS3                
    model.addConstrs((quicksum(y0[s][i, j, k] for i,j in A if j == jj) - quicksum(y0[s][i, j, k] for i,j in A if i == jj) 
            == 0 for k in K for jj in Nint for s in S), name='(FS3)')
    ## SECOND STAGE CONSTRAINTS ##
    #SS1
    model.addConstrs((quicksum(y[s][i, j, k] for k in K) 
            <= 1 for i,j in AF for s in S), name='(SS1)')
    #SS2        
    model.addConstrs((y[s][i, j, k] == y0[s][i, j, k]
            + zplus[s][i, j, k] - zminus[s][i, j, k] for k in K for i,j in A for s in S), name="(SS2)")
    #SS3
    model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if i == ii) 
            == nav[ii, k] for k in K for ii in Nf for s in S), name='(SS3)')        
    #SS4
    model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if j == jj) - quicksum(y[s][i, j, k] for i,j in A if i == jj) 
            == 0 for k in K for jj in Nint for s in S), name='(SS4)')
    #SS5
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
    # #SS6
    # model.addConstrs((quicksum(size[s][item]*x[s][i, j, item] for item in Cargo)
    #         <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS6)")
    # #SS7
    # model.addConstrs((quicksum(vol[s][item]*x[s][i, j, item] for item in Cargo)
    #         <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF for s in S), name="(SS7)")
    #SS8
    model.addConstrs((x[s][i, j, item] 
            <= quicksum(y[s][i, j, k] for k in K) for item in Cargo for i,j in AF for s in S), name="(SS8)")
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
            <= y0[s][i, j, k] for k in K for i,j in AF for s in S), name="(SS12)")  
    #SS13
    model.addConstrs((y0[s][i, j, k] + zplus[s][i, j, k] 
            <= 1 for k in K for i,j in AF for s in S), name="(SS13)")             
    #z+_bin_flight
    model.addConstrs((zplus[s][i, j, k] 
            <= 1 for k in K for i,j in AF for s in S), name="(z+_bin_flight)")
    #z-_bin_flight
    model.addConstrs((zminus[s][i, j, k] 
            <= 1 for k in K for i,j in AF for s in S), name="(z-_bin_flight)")        
    # Retiming       
    #RT1
    model.addConstrs((y0[s][i, j, k] + zminus[s][i, j, k] + zplus[s][i2, j2, k]
            <= 2 + r[s][i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j] for s in S), name="(RT1)")		        
    #RT2
    model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= y0[s][i, j, k] for k in K for i,j in AF for s in S), name="(RT2)")		        
    #RT3
    model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
            <= zminus[s][i, j, k] for k in K for i,j in AF for s in S), name="(RT3)")      
    #RT4
    model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i,j in V[i2, j2])
            <= zplus[s][i2, j2, k] for k in K for i2,j2 in AF for s in S), name="(RT4)")            
                        
    ##########################
    ### Objective Function ###
    ##########################
    # Definition of the second stage value        
    SSP = {}  #{s: SSPs} second stage profit for each scenario
    TRC = {}  #{s: TRCs} total recovery cost
    RC = {}  #{s: RCs} re-timing costs
    for s in S:
        SSP[s] = (  quicksum(inc[s][item]*q[s][item] for item in Cargo)
                - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, k)]*size[s][item]*w[s][arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                - quicksum(ch[arco]*size[s][item]*x[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
        TRC[s] = quicksum( sc[s][(arco, k)]*zplus[s][arco[0], arco[1], k] for k in K for arco in AF)

        RC[s] = quicksum(tv*gap[arco, arco2]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k]
                        - sc[s][arco2, k]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])
    Q = quicksum(SSP[s] - TRC[s] - RC[s] for s in S)/len(S)            
    obj = Q
    model.setObjective(obj, GRB.MAXIMIZE)
    model.Params.Threads = 1
    if L_time != None:
        model.setParam('TimeLimit', L_time)
    if L_gap != None:
        model.setParam('MIPGap', L_gap)        
    model.update()
    model.optimize()
    run_time = time.time() - start_time
    if False:
        print("vuelos en primera etapa: ", sum(y0[0][i, j, k].X for k in K for i,j in AF))
        for s in S:
            print("Scenario ", s)
            ssp, trc, rc  = SSP[s].getValue(), TRC[s].getValue(), RC[s].getValue()
            valor = ssp - trc - rc
            print(" Q[s] = SSP[s] - TRC[s] - RC[s] = {:.0f} - {:.0f} - {:.0f} = {:.0f}".format(ssp, trc, rc, valor))
            print(" vuelos en descartados en segunda etapa: ", sum(zminus[s][i, j, k].X for i,j in AF for k in K))
            print(" vuelos en agregados en segunda etapa: ", sum(zplus[s][i, j, k].X for i,j in AF for k in K))
    print("****L = {} computed in {}s****".format(model.ObjBound, run_time))

    return model.objVal, model.ObjBound, model.status




if __name__ == '__main__':
    print('Module to create an upper bound L by solving a relaxed similar problem')
