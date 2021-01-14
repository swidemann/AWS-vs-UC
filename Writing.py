#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

import os
from gurobipy import GRB
import statistics as st

def write_model(directory, name, model):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/" + name
    model.write(path)


def write_log(directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/" + name
    os.rename(name, path)


def write_parameters(instancia, days, S, airports, K, n, Cargo, N, A, AF, directory, name, model, q, size, FS=False, RR =False):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/" + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        file.write("PARAMETERS\n")
        file.write("Days: {}\n".format(days))
        file.write("Scenarios #: {}\n".format(len(S)))
        file.write("Airports #: {}\n".format(len(airports)))
        file.write("Aircraft types #: {}\n".format(len(K)))
        sum = 0
        for k in K:
            sum += len(n[k])
        file.write("Aircraft total #: {}\n".format(sum))
        file.write("ODs #: {}\n".format(len(Cargo)))
        file.write("Nodes #: {}\n".format(len(N)))
        file.write("Arcs #: {}\n".format(len(A)))
        file.write("-> Flight Arcs #: {}\n".format(len(AF)))
        file.write("Flight variable #: |AF|x|K| = {}\n".format(len(AF)*len(K)))
        if model.status != GRB.Status.INFEASIBLE:
            file.write("SOLUTION\n")
            file.write("Constraints #: {}\n".format(model.NumConstrs))
            file.write("Variables #: {}\n".format(model.NumVars))
            file.write("Best objective: {}\n".format(model.ObjVal))
            file.write("Time: {}\n".format(model.Runtime))
            solutions = [0]
            if not FS:
                solutions += S
            for scenario in solutions:
                available_ods = 0
                transported_ods = 0
                for item in Cargo:
                    if size[scenario][item] > 0:
                        available_ods += 1
                    if (not FS and scenario != 0) or FS or RR:
                        transported_ods +=int(q[scenario][item].x)
                    else:
                        transported_ods +=int(q[scenario][item])
                file.write("Available ODs in scenario {}: {}\n".format(scenario, available_ods))
                file.write("Transported ODs in scenario {}: {}\n".format(scenario, transported_ods))


def write_runtime(directory, name, time_name, time_value, new_file = False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = directory + "/" + name
    if not ".txt" in path:
        path += ".txt"
    
    mode = "a"  # append
    if new_file:
        mode = "w"  # write from scratch
    with open(path, mode) as file:
        file.write("{}: {}\n".format(time_name, time_value))


def write_ods(directory, name, Cargo, OD, S, size, created_cargo_types, inc):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = directory + "/" + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        file.write("ODs GENERATED:\n")
        for item in Cargo:
            file.write("\nOD: {}, {}, type {}\n".format(item, OD[item], created_cargo_types[item]))
            file.write("Base size: {:.0f}, inc: {:.0f}\n".format(size[0][item], inc[0][item]))
            for s in S:
                file.write("-> in scenario {}: {:.0f}, inc: {:.0f}\n".format(s, size[s][item], inc[s][item]))
                if size[0][item] > 0:
                    var = (size[s][item]-size[0][item])/size[0][item]*100
                    file.write("--> variation: {:.0f}%\n".format(var))


def clear_cuts(directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Cuts " + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        pass      


def write_cuts(directory, name, FSC, theta_tongo, Q_calculado, tiempo, integer=False, QT_bound=False, comment=""):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Cuts " + name
    if not ".txt" in path:
        path += ".txt"

    line = "FSC(y^): {:10.2f}, Theta^: {:10.2f}, E[Q(y^)]: {:10.2f}, Cut time: {:5.5f}s".format(FSC, theta_tongo, Q_calculado, tiempo)
    if integer:
        line = line + " INTEGER"
    if QT_bound:
        line = line + " QTbound"
    if comment:
        line = line + " " + comment
    line = line + "\n"
    with open(path, "a") as file:
        file.write(line)


def write_cuts_forecast(directory, name, FSC, Q_s1, theta_tongo, Q_calculado, tiempo, integer=False):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Cuts " + name
    if not ".txt" in path:
        path += ".txt"

    line = "FSC(y^): {:10.2f}, Q_s1: {:10.2f}, Theta^: {:10.2f}, E[Q(y^)]: {:10.2f}, Cut time: {:5.5f}s".format(FSC, Q_s1, theta_tongo, Q_calculado, tiempo)
    if integer:
        line = line + " INTEGER\n"
    else:
        line = line + "\n"

    with open(path, "a") as file:
        file.write(line)        


def clear_bounds(directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Bound " + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        pass     


def TEST_write_QTbound(directory, name, Q_real, QTbound, T_cost, Q_previous, time):
    """
    from a new solution, the best bound calculated was T_cost+Q_previous from a previously visited solution
    the real value of this new solution is Q_real
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Bound " + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "a") as file:
        percentage = QTbound/Q_real*100
        line = "Q(y1): {:10.2f}, QTbound(y1): {:10.2f} ({:3.2f}%), T(y2,y1): {:10.2f}, Q(y2): {:10.2f}, Time: {:5.5f}s".format(Q_real, QTbound, percentage, T_cost, Q_previous, time)
        if QTbound < Q_real:
            line += " INVALID\n"
        else:
            line += "\n"
        file.write(line)


def write_orders(directory, name, orders, details=False):
    # orders = {}, sol1_key: {sol1_key: [Q(sol1), sol1_dict], sol2_key: Q(sol2_key), ...}  sol1 representative, sol2 in order with
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory + "/Order " + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        for key in orders.keys():
            Q_vals = []      
            line = "Q(rep): {:10.2f}".format(orders[key][key][0])
            Q_vals.append(orders[key][key][0])
            for key2 in orders[key].keys():
                if key2 != key:
                    line += ", {:10.2f}".format(orders[key][key2])
                    Q_vals.append(orders[key][key2])
            line += "\n"
            file.write(line)
            sd = 0
            if len(Q_vals) > 1:
                sd = st.stdev(Q_vals)
            line2 = " av: {:10.2f}, sd: {:.2f}\n".format(sum(Q_vals)/len(Q_vals), sd)
            file.write(line2)


def __streamline_schedule(nodes):
    nodes.sort(key=lambda x: int(x[3:]))  # sort by the hour of the nodes, ex "BBB17" before "AAA23"
    to_write = []
    for i in range(len(nodes)):
        if i == 0 or i == len(nodes)-1:
            to_write.append(nodes[i])
        else:
            prev_airport = nodes[i-1][0]
            this_airport = nodes[i][0]
            next_airport = nodes[i+1][0]
            if prev_airport != this_airport or this_airport != next_airport:
                to_write.append(nodes[i])
    schedule = "   " + " -> ".join(to_write)
    return schedule


def write_cargo_schedule(directory, name, A, S, Cargo, OD, ex, x):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + "/" + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        file.write("\n### CARGO SCHEDULE ###\n")
        for s in S:
            file.write("\n")
            file.write("\n## Scenario {} ##\n".format(s))
            for item in Cargo:
                file.write("\n# Item {}: {} #\n".format(item, OD[item]))
                if ex[s][item] == 1:
                    nodes = []
                    for i,j in A:
                        if x[s][i, j, item].x > 0.5:
                            if i not in nodes:
                                nodes.append(i)
                            if j not in nodes:
                                nodes.append(j)
                    schedule = __streamline_schedule(nodes)
                    file.write(schedule)
                else:
                    file.write("  doesn't exists in this scenario")


def write_retimings(directory, name, S, AF, V, K, y, r, zplus, zminus):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + "/" + name
    if not ".txt" in path:
        path += ".txt"

    with open(path, "w") as file:
        file.write("\n### RETIMINGS ###\n")
        for s in S:
            file.write("\n")
            file.write("\n## Scenario {} ##\n".format(s))
            for i,j in AF:
                for i2,j2 in V[i,j]:
                    for k in K:
                        if r[s][i, j][i2, j2, k].x > 0.5:
                            file.write("\nRetiming from {}->{} to {}->{} with aircraft type {}".format(i, j, i2, j2, k))
                            file.write("\n y[0][{}, {}, {}]     exists: {}".format(i, j, k, y[0][i, j, k].x))
                            file.write("\n zminus[{}][{},{},{}] discarded: {} ".format(s, i, j, k, zminus[s][i, j, k].x))
                            file.write("\n zplus[{}][{},{},{}]  created: {} ".format(s, i2, j2, k, zplus[s][i2, j2, k].x))
                  


if __name__ == "__main__":
    print("Modulo con funciones para escribir archivos de output")