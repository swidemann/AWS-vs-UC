#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

from gurobipy import Model, GRB, quicksum
from Printing import print_aggregated_fs_TEST
from Improvements_functions import check_best_bound
import math

lagThreads = 1


def compute_rho_idea1(y0_param, A, N, AF, AG, K, airports, int_sol_hash, int_Q_bound_hash, sc, V, gap, tv, S, Q_int_calculated, L):
    othery0s_dist1 = create_HammingDistance_1(y0_param, A, N, AF, AG, K, airports)
    M = 0
    for other_y0 in othery0s_dist1:
        QT_bound_lagrangian =  check_best_bound(other_y0, int_sol_hash, int_Q_bound_hash, sc, V, gap, tv, AF, K, S) 
        if QT_bound_lagrangian > M:
            M = QT_bound_lagrangian
    # M - Q(y^) <= rho <= L - Q(y^)
    b1 = M - Q_int_calculated
    b2 = L - Q_int_calculated 
    if b1 <= b2 and M != math.inf:
        rho = b1
    else:
        print("### ERROR LAGRANGIAN CUT: M - Q(y^) <= rho <= L - Q(y^) NOT SATISFIED ###")
        rho = None
    return rho



def create_HammingDistance_1(y0_param, A, N, AF, AG, K, airports):
    """
    Delta(y0_param, other_y0) = 1 -> either removing a last flight, or adding a flight at the end
    """
    TEST = False
    other_y0s = []
    # removing last flight
    if TEST:
        input("\nITERACION NUEVA[enter]")
        cont_y0_nuevo = 0
        print("y0_param:")
        for k in K:
            print_aggregated_fs_TEST(airports, N, A, 24, k, y0_param, f"y0param")
    for i,j in AF:
        for k in K:
            air_dep = i[:3]
            hour_dep = int(i[3:])
            air_arriv = j[:3]
            hour_arriv = int(j[3:])    
            y0_nuevo = {}
            valid = True
            if y0_param[i,j,k] == 1:  # vuelo a descartar
                for i2,j2 in AF:
                    y0_nuevo[i2,j2,k] = y0_param[i2,j2,k] if (i,j) != (i2,j2) else 0  # descartar el vuelo
                for i2,j2 in AG:
                    if i2[:3] == air_dep:  # aircraft didnt left
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]+1 if int(j2[3:]) > hour_dep else y0_param[i2,j2,k]
                    elif i2[:3] == air_arriv:  # aircraft didint arrive
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]-1 if int(j2[3:]) > hour_arriv else y0_param[i2,j2,k]
                    else:
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]
            else:  # y0_param[i,j,k] == 0, vuelo a agregar
                for i2,j2 in AF:
                    y0_nuevo[i2,j2,k] = y0_param[i2,j2,k] if (i,j) != (i2,j2) else 1  # agregar el vuelo
                for i2,j2 in AG:
                    if i2[:3] == air_dep:  # aircraft left
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]-1 if int(j2[3:]) > hour_dep else y0_param[i2,j2,k]
                    elif i2[:3] == air_arriv:  # aircraft arrive
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]+1 if int(j2[3:]) > hour_arriv else y0_param[i2,j2,k]
                    else:
                        y0_nuevo[i2,j2,k] = y0_param[i2,j2,k]  
            # check validity of new schedule 
            if TEST:
                cont_y0_nuevo += 1
                print(" y0_nuevo creado:")
            for i2,j2 in AF:
                if y0_nuevo[i2,j2,k] == 1 and valid==True:  # flight in the new schedule and schedule not discarded
                    hour_dep2 = int(i2[3:])
                    if hour_dep2 >= hour_dep:  # flight after the departure of the descarded/added flight
                        # flight must still be feasible: flow arriving to departure node must be > 0
                        if sum(y0_nuevo[i3,j3,k] for i3,j3 in A if j3==i2) > 0:
                            pass
                        else:
                            valid = False
            if valid:
                other_y0s.append(y0_nuevo)
            if TEST:
                name = f"y0nuevo#{cont_y0_nuevo}_VALIDO" if valid else f"y0nuevo#{cont_y0_nuevo}_INVALIDO"
                print_aggregated_fs_TEST(airports, N, A, 24, k, y0_nuevo, name)
                print(f"  y0_nuevo {cont_y0_nuevo} valido" if valid else f"  y0_nuevo {cont_y0_nuevo} INVALIDO")
    
    return other_y0s



def create_alpha_vector(rho, y0_param, AF, K):
    alpha_vector = {}
    for i,j in AF:
        for k in K:
            signo = -1 if y0_param[i,j,k] == 1 else 1
            alpha_vector[i,j,k] = float(rho)*signo

    return alpha_vector
    


def solve_beta_s(alpha_vector, s, beta_qTy, beta_sat, y0z_var, AF, K):
    #update objective function
    qTy_minus_alphaTz = beta_qTy[s] - sum(alpha_vector[i,j,k]*y0z_var[s][i,j,k] for i,j in AF for k in K)
    beta_sat[s].setObjective(qTy_minus_alphaTz, GRB.MAXIMIZE)
    #solve model
    beta_sat[s].optimize()
    return beta_sat[s].ObjBound



def create_second_stage_beta(days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, sat_gap, integer, disconnected=None, limited_actions=None):
    ########################
    ### Satellite Models ###
    ########################
    models = {}  #{s: model_s}
    qTy = {}  #{s: objective function without alphaTy}  y generic second stage variables
    y0z_var = {}  #{s: y0z variables in model_s}

    for s in S:
        model = Model("beta d{}s{} scenario {}".format(days, len(S), s))

        #################
        ### Variables ###
        #################
        y = {}  #{s: y[i, j, k]}
        # second stage
        x = {}  #{s: {x[i, j, od]}}
        q = {}  #{s: {q[od]}
        w = {}  #{s: {w[i, j, od, k]}}    
        zplus = {}  #{s: {k: zplus[arco, m]}}
        zminus = {}  #{s: {k: zminus[arco, m]}}
        y[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='y{}'.format(s))
        zplus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zplus{}'.format(s))
        zminus[s] = model.addVars(A, K, vtype=GRB.INTEGER, name='zminus{}'.format(s))
        x[s] = model.addVars(A, Cargo, vtype=GRB.BINARY, name='x{}'.format(s))
        w[s] = model.addVars(AF, Cargo, K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='w{}'.format(s))
        q[s] = model.addVars(Cargo, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='q{}'.format(s))
        #retiming
        r = {}  # {s: {arco: r[arco2, k]}},  r[s][arco][V[arco], k]
        rs = {}
        for arco in AF:
            rs[arco] = model.addVars(V[arco], K, vtype=GRB.CONTINUOUS, ub=1.0, lb=0.0, name='r({},{}){}'.format(arco[0], arco[1], s))
        r[s] = rs
        #new variable for lagrangian cut
        #y0z[i, j, k] = y[0][i, j, k] but z in general lagrangian formulation (x = z on Q(x))
        if integer:
            y0z = model.addVars(A, K, vtype=GRB.INTEGER, name='y0z')
        else:
            y0z = model.addVars(A, K, vtype=GRB.CONTINUOUS, name='y0z')

        ###################
        ### Constraints ###
        ###################
        # FIRST STAGE CONSTRAINTS FOR y0z ##
        #FS1
        model.addConstrs((quicksum(y0z[i, j, k] for k in K) 
                <= 1 for i,j in AF ), name='(FSy0z1)')
        #FS2
        model.addConstrs((quicksum(y0z[i, j, k] for i,j in A if i == ii) 
                == nav[ii, k] for k in K for ii in Nf ), name='(FSy0z2)') 
        #FS3                
        model.addConstrs((quicksum(y0z[i, j, k] for i,j in A if j == jj) - quicksum(y0z[i, j, k] for i,j in A if i == jj) 
                == 0 for k in K for jj in Nint), name='(FS3)')
        # SECOND STAGE CONSTRAINTS ##
        #SS1
        model.addConstrs((quicksum(y[s][i, j, k] for k in K) 
                <= 1 for i,j in AF ), name='(SS1)')
        #SS2        
        model.addConstrs((y[s][i, j, k] == y0z[i, j, k]
                + zplus[s][i, j, k] - zminus[s][i, j, k] for k in K for i,j in A ), name="(SS2)")
        #SS3
        model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if i == ii) 
                == nav[ii, k] for k in K for ii in Nf ), name='(SS3)')        
        #SS4
        model.addConstrs((quicksum(y[s][i, j, k] for i,j in A if j == jj) - quicksum(y[s][i, j, k] for i,j in A if i == jj) 
                == 0 for k in K for jj in Nint ), name='(SS4)')
        #SS5
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
        #SS6
        model.addConstrs((quicksum(size[s][item]*x[s][i, j, item] for item in Cargo)
                <= quicksum(air_cap[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS6)")
        #SS7
        model.addConstrs((quicksum(vol[s][item]*x[s][i, j, item] for item in Cargo)
                <= quicksum(air_vol[k]*y[s][i, j, k] for k in K) for i,j in AF ), name="(SS7)")
        #SS8
        model.addConstrs((x[s][i, j, item] 
                <= quicksum(y[s][i, j, k] for k in K) for item in Cargo for i,j in AF ), name="(SS8)")
        #SS9 
        model.addConstrs((x[s][i, j, item] + y[s][i, j, k] 
                <= w[s][i, j, item, k] + 1 for k in K for item in Cargo for i,j in AF ), name="(SS9)")        
        #SS10
        # force mandatory Cargo only when not limited recovery actions
        if limited_actions == None:
            model.addConstrs((q[s][item] 
                    >= mand[s][item] for item in Cargo ), name="(SS10)")
        #SS11
        model.addConstrs((q[s][item] 
                <= ex[s][item] for item in Cargo ), name="(SS11)")
        #SS11
        model.addConstrs((zminus[s][i, j, k] 
                <= y0z[i, j, k] for k in K for i,j in AF), name="(SS11)")  
        #SS12
        model.addConstrs((y0z[i, j, k] + zplus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(SS12)")                 
        #z+_bin_flight
        model.addConstrs((zplus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(z+_bin_flight)")
        #z-_bin_flight
        model.addConstrs((zminus[s][i, j, k] 
                <= 1 for k in K for i,j in AF), name="(z-_bin_flight)")        
        # Retiming       
        #RT1
        model.addConstrs((y0z[i, j, k] + zminus[s][i, j, k] + zplus[s][i2, j2, k]
                <= 2 + r[s][i, j][i2, j2, k] for k in K for i,j in AF for i2,j2 in V[i, j] ), name="(RT1)")		        
        #RT2
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
                <= y0z[i, j, k] for k in K for i,j in AF), name="(RT2)")		        
        #RT3
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i2,j2 in V[i, j])
                <= zminus[s][i, j, k] for k in K for i,j in AF), name="(RT3)")      
        #RT4
        model.addConstrs((quicksum(r[s][i, j][i2, j2, k] for i,j in V[i2, j2])
                <= zplus[s][i2, j2, k] for k in K for i2,j2 in AF), name="(RT4)") 

        if limited_actions != None:
            model.addConstr(quicksum(zplus[s][i,j,k] + zminus[s][i,j,k] for k in K for i,j in AF) <= limited_actions)           

        ##########################
        ### Objective Function ###
        ##########################
        # Definition of the second stage value        
        SSP = {}  #{s: SSPs} second stage profit for each scenario
        TRC = {}  #{s: TRCs} total recovery cost
        RC = {}  #{s: RCs} re-timing costs
        SSP[s] = (  quicksum(inc[s][item]*q[s][item] for item in Cargo)
                - quicksum(cf[(arco, k)]*y[s][arco[0], arco[1], k] for arco in AF for k in K)
                - quicksum(cv[(arco, k)]*size[s][item]*w[s][arco[0], arco[1], item, k] for item in Cargo for arco in AF for k in K)
                - quicksum(ch[arco]*size[s][item]*x[s][arco[0], arco[1], item] for item in Cargo for arco in AG) 
                )
        TRC[s] = quicksum( sc[s][(arco, k)]*zplus[s][arco[0], arco[1], k] for k in K for arco in AF)

        RC[s] = quicksum(tv*gap[arco, arco2]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k]
                        - sc[s][arco2, k]*r[s][arco[0], arco[1]][arco2[0], arco2[1], k] for k in K for arco in AF for arco2 in V[arco])

        obj = SSP[s] - TRC[s] - RC[s]  # qTy
        model.setObjective(0, GRB.MAXIMIZE)
        model.setParam('OutputFlag', False)
        if disconnected != None:
            model.Params.Disconnected = disconnected
        model.Params.Threads = lagThreads
        # model.setParam("LogFile", "log {} {} threads.log".format(model.ModelName, model.Params.Threads))
        if sat_gap != None:
            model.setParam('MIPGap', sat_gap)
        model.update()

        models[s] = model
        qTy[s] = obj
        y0z_var[s] = y0z

    return models, qTy, y0z_var






if __name__ == "__main__":
    print("Modulo con funciones para cortes lagrangeanos")