#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:27:17 2020

@author: steffan
"""

import math
import Printing as prt
from Printing import print_aggregated_fs_TEST
import itertools


def create_FullOrder(y0_param, A, N, AF, AG, K, airports, nav):
    """
    para cada y0_param, retorna un conjunto con soluciones que cumplen la relacion de orden y algunas que no
    para cada vuelo de y0_param crea su V(ij) y luego todas las permutaciones entre ellas para generar soluciones nuevas
    esto NO genera todas las que satisfacen la relacion de orden ya que cada V(ij) es dependiente de y0_param
    esto genera algunas soluciones que NO satisfacen la relacion de orden, ya que hacer las permutaciones puede generar swaps y cosas que no cumplan
    esto no es un problema ya que la cota QT es valida para cualquier par de soluciones, no necesitan la relacion de orden
    """
    TEST = False
    other_y0s = []  # soluciones a retornar
    conjuntos_vuelos = {k: [] for k in K}  # {k: [ listas de de V(ij)U{ij} para cada ij operado por k]}
    # crear los similares a cada vuelo operado
    if TEST:
        input("\nITERACION NUEVA[enter]")
        print("y0_param:")
        for k in K:
            print_aggregated_fs_TEST(airports, N, A, 24, k, y0_param, f"y0param")
        print(f"conjuntos vuelos: {conjuntos_vuelos}")
    for i,j in AF:
        for k in K:
            if y0_param[i,j,k] == 1:  # vuelo existe
                if TEST:
                    print(f"tipo {k}: {i}{j}")
                Similar = generate_similar_flights(y0_param, i, j, k, AF, airports, K)
                similares_y_originales = [(i2,j2) for i2,j2 in Similar]
                similares_y_originales.append( (i,j) )
                conjuntos_vuelos[k].append(similares_y_originales)
    if TEST:
        print(f"conjuntos vuelos: {conjuntos_vuelos}")
        for k in K:
            print(f"vuelos similares aviones tipo {k}:")
            for vuelos in conjuntos_vuelos[k]:
                # print(f"conjunto de {vuelos[-1]}")
                print(f" vuelos: {vuelos}")
    # generar las combinaciones sobre V(ij) para cada ij operado por k
    combinaciones = {k: list(itertools.product(*conjuntos_vuelos[k])) }  #*a= desempacar los elementos de a
    # combinaciones[k] = lista con cada elemento uno de los del producto de [V(ij) para todo ij operado por k]
    # los elementos son un grupo de arcos ij in AF que (potencialmente) es una solucion
    if TEST:
        for k in K:
            print(f"Variaciones para tipo {k}:")
            for vuelos in combinaciones[k]:
                print(f" itinerario nuevo: {vuelos}")
    if len(K) > 1:
        print("ERROR EN Functions_impovements.py, create_FullOrder no implementado para |K|>1")
        raise Exception
    # generar las soluciones
    AG_ordenados = [(i,j) for i,j in AG]
    AG_ordenados.sort(key=lambda x: int(x[0][3:]))  # ordenar por las horas iniciales
    cont_nuevos = 1
    for k in K:
        for vuelos in combinaciones[k]:
            y0_nuevo = {}
            valido = True
            for i,j in AF:  # arcos de vuelo
                for k in K:
                    y0_nuevo[i,j,k] = 1 if (i,j) in vuelos else 0
            for i,j in AG_ordenados:
                if int(i[3:]) == 0:  # primer arco de tierra AAA0,AAAt
                    for k in K:
                        y0_nuevo[i,j,k] = nav[i, k]
                else:  # otros arcos de tierra
                    for k in K:
                        if valido:  # si no se ha descartado aun
                            cantidad_en_arco = sum(y0_nuevo[i2,j2,k] for i2,j2 in A if j2==i)  # arcos que llegan
                            cantidad_en_arco -= sum(y0_nuevo[i2,j2,k] for i2,j2 in AF if i2==i)  # vuelos que salen
                            if cantidad_en_arco < 0:
                                valido = False
                            y0_nuevo[i,j,k] = cantidad_en_arco
            if valido and y0_nuevo != y0_param:  # guardar si es valido y no es el original
                other_y0s.append(y0_nuevo)
            if TEST and y0_nuevo != y0_param:
                name = f"y0nuevo#{cont_nuevos}_VALIDO" if valido else f"y0nuevo#{cont_nuevos}_INVALIDO"
                print_aggregated_fs_TEST(airports, N, A, 24, k, y0_nuevo, name)
                print(f"  y0_nuevo {cont_nuevos} valido" if valido else f"  y0_nuevo {cont_nuevos} INVALIDO")
                cont_nuevos += 1
    return other_y0s





def check_best_bound(y0_param, int_sol_hash, int_Q_bound_hash, sc, V, gap, tv, AF, K, S):
    """
    from a new solution y0_param, get min(T(y02, y0_param) + Q(y02) for y02 previously checked)
    Q(y02) >= -T(y02, y0_param) + Q(y0_param) -> Q(y0_param) <= Q(y02) + T(y02, y0_param)
    """
    best_bound = math.inf
    for y0_2_key in int_sol_hash.keys():  # previously visited solutions
        y0_2_dict =  int_sol_hash[y0_2_key]
        # cost from transforming the old solution into the new one
        T_cost_old_into_new = calculate_transf_cost(y0_2_dict, y0_param, sc, V, gap, tv, AF, K, S)
        Q_y02 = int_Q_bound_hash[y0_2_key]
        this_bound = T_cost_old_into_new + Q_y02
        if this_bound < best_bound:
            best_bound = this_bound
    return best_bound



def check_order(y0_1, y0_2, airports, days, AF, K):
    """
    check if a first stage solution y0_1 has an order relation with another y0_2
    (idea: for a schedule, another schedule thats the same but translated 1 hour should be the same)
    doesnt allow departure swap
    """

    vuelos1 = {}  # stack por origen destino tipo de avion
    vuelos2 = {}
    for air1 in airports:
        for air2 in airports:
            if air1 != air2:
                for k in K:  # armar stack
                    vuelos1[air1, air2, k] = []
                    vuelos2[air1, air2, k] = []

    # order flights per day, and nights on edge cases
    flights_perday_1, flights_perday_2 = {}, {}
    night_flights_1, night_flights_2 = {}, {}
    days_list = list(range(1, days+1))
    for k in K:
        for d in days_list:
            flights_perday_1[d, k], flights_perday_2[d, k] = [], []
            if d+1 in days_list:
                night_flights_1[d, d+1, k], night_flights_2[d, d+1, k] = [], []
    for k in K:
        for i,j in AF:
            if y0_1[i,j,k] == 1:
                h_departure = int(i[3:])
                h_arrival = int(j[3:])
                d_departure = h_departure//24 +1
                d_arrival = h_arrival//24 +1
                if h_arrival%24 == 0:  # llegar justo a las 12PM es dentro del dia
                    d_arrival -= 1
                if d_departure == d_arrival:  # day flight
                    flights_perday_1[d_departure, k].append((i,j))
                else:
                    night_flights_1[d_departure, d_arrival, k].append((i,j))
                vuelos1[i[:3], j[:3], k].append((i,j))  # agregar al stack
            if y0_2[i,j,k] == 1:
                h_departure = int(i[3:])
                h_arrival = int(j[3:])
                d_departure = h_departure//24 + 1
                d_arrival = h_arrival//24 + 1
                if h_arrival%24 == 0:  # llegar justo a las 12PM es dentro del dia
                    d_arrival -= 1
                if d_departure == d_arrival:  # day flight
                    flights_perday_2[d_departure, k].append((i,j))
                else:
                    night_flights_2[d_departure, d_arrival, k].append((i,j))
                vuelos2[i[:3], j[:3], k].append((i,j))  # agregar al stack
    # first: check if for each OD (A->B) theres the same number of flights for every day and night
    for air1 in airports:
        for air2 in airports:
            if air1 != air2:
                for d in days_list:
                    for k in K:
                        flights_1 = sum(1 for i,j in flights_perday_1[d, k] if i[:3]==air1 and j[:3]==air2)
                        flights_2 = sum(1 for i,j in flights_perday_2[d, k] if i[:3]==air1 and j[:3]==air2)
                        if flights_1 != flights_2:
                            return False  # for a day, have a different schedule
                        if d+1 in days_list:
                            flights_1 = sum(1 for i,j in night_flights_1[d, d+1, k] if i[:3]==air1 and j[:3]==air2)
                            flights_2 = sum(1 for i,j in night_flights_2[d, d+1, k] if i[:3]==air1 and j[:3]==air2)
                            if flights_1 != flights_2:
                                return False  # for a night, have a different schedule
                        # sort stacks
                        vuelos1[air1, air2, k].sort(key=lambda x: int(x[1][3:]))
                        vuelos2[air1, air2, k].sort(key=lambda x: int(x[1][3:]))                                      
    # second: check order
    # para cada par air1 air2, para cada vuelo, para cada air3 != air1, la cantidad de vuelos air3->air1 que llegan antes de la salida de vuelo1 debe ser igual
    # stack para cada solucion para cada OD, despues para cada OD observar los vuelos stack1[od][i] y stack2[od][i] y ahi ver las vuelos previos
    for air1 in airports:
        for air2 in airports:
            if air1 != air2:
                for k in K:
                    for ind in range(len(vuelos1[air1, air2, k])-1, -1, -1):  # recorrer stack desde el final hasta el comienzo
                        v1 = vuelos1[air1, air2, k][ind]
                        h_departure1 = int(v1[0][3:])  # hora salida del vuelo en sol1
                        v2 = vuelos2[air1, air2, k][ind]
                        h_departure2 = int(v2[0][3:])  # hora salida del vuelo en sol2
                        for air3 in airports:
                            if air3 != air1:  
                                # vuelos que LLEGAN al origen del vuelo observado antes de que salga
                                arrivals_1 = sum(1 for i,j in vuelos1[air3, air1, k] if int(j[3:])<=h_departure1)
                                arrivals_2 = sum(1 for i,j in vuelos2[air3, air1, k] if int(j[3:])<=h_departure2)
                                # print("sol1: vuelo {}, llegadas previas: {}".format(v1, flights_1))
                                # print("sol2: vuelo {}, llegadas previas: {}".format(v2, flights_2))
                                if arrivals_1 != arrivals_2:
                                    return False
                                # vuelos que SALEN del origen del vuelo observado antes de que salga
                                departures_1 = sum(1 for i,j in vuelos1[air1, air3, k] if int(i[3:])<=h_departure1)
                                departures_2 = sum(1 for i,j in vuelos2[air1, air3, k] if int(i[3:])<=h_departure2)                                
                                if departures_1 != departures_2:
                                    return False                                
    # si paso todos los requisitos previos
    return True



def calculate_transf_cost(y0_2, y0_1, sc, V, gap, tv, AF, K, S):
    """
    Calculate T(y02, y01): how costly is to transform old solution y02 into new solution y01
    Discard every flight that is in y0_2 an is not in y0_1 (at cost 0)
    and add every flight that is in y0_1 an is not in y0_2 (at cost sc)
    BY default NOT checking retimings (at cost tv) inside vecinity V (could allow retiming in T and also Q)
    """
    print_details = False

    to_add = {}  # {k: [flights in y0_1 and not in y0_2]}
    to_remove = {}  # {k: [flights not in y0_1 and in y0_2]}

    for k in K:
        to_add[k] = []
        to_remove[k] = []

    for k in K:
        for i,j in AF:
            if y0_2[i,j,k] == 0 and y0_1[i,j,k] == 1:  # must be added
                to_add[k].append( (i,j) )
            if y0_2[i,j,k] == 1 and y0_1[i,j,k] == 0:  # must be discarded
                to_remove[k].append( (i,j) )
    if print_details:
        for k in K:
            print("type {}:".format(k))
            print(" must remove: ", to_remove[k])
            print(" must add: ", to_add[k])
        print("")
    # calculate cost
    total_cost = 0
    for k in K:
        if print_details:
            print("type {}:".format(k))
        for (i2,j2) in to_add[k]:
            cost = sum(sc[s][(i2, j2), k] for s in S)/len(S)
            total_cost += cost
            if print_details:
                print(" cost added for adding {}: {}".format( (i2,j2), cost ))
    if print_details:
        input("press enter...")
    return total_cost



def generate_similar_flights(y, i, j, k, AF, airports, K):
    """
    for a given solution y, focusing on a specific ij flight operated by a k type aircraft (y[i,j,k]=1)
    return a list of feasible FLIGHTS that can replace the observed flight
    the list doesnt include the original flight ij
    """
    air1 = i[:3]
    air2 = j[:3]
    h_departure = int(i[3:])
    h_arrival = int(j[3:])

    #RULE 0 chained flights
    # si otro(s) vuelo sale del mismo nodo del que sale el vuelo observado, entonces no se puede mover
    if sum(y[i2,j2,k] for k in K for i2,j2 in AF if i2 == i and j2 != j) > 0:
        return []

    # Cargo related time bounds
    d_departure = h_departure//24 +1
    d_arrival = h_arrival//24 +1
    if h_arrival%24 == 0:  # llegar justo a las 12PM es dentro del dia
        d_arrival -= 1
    dep_lower_cargo = (d_departure-1)*24
    dep_upper_cargo = math.inf  # bound for night flights
    arriv_lower_cargo = 0  # bound for night flights
    arriv_upper_cargo = d_arrival*24
    if d_departure != d_arrival:  # if its a night flight, the alternatives must alse be night flights
        # d_departure*24 is the midnight of the flight
        dep_upper_cargo = d_departure*24 - 1
        arriv_lower_cargo = d_departure*24 + 1

    # DEPARTURE LOWER BOUND
    #RULE 1 previous Cargo
    R1_bound = [dep_lower_cargo]
    #RULE 2 arrivals to the same origin airport                                                
    R2_bound = [ int(j2[3:]) for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                and j2[:3] == air1 
                                                and int(j2[3:]) <= h_departure ]
    #RULE 3 departures from the same origin airport STRICT INEQUALITY
    R3_bound = [ int(i2[3:])+1 for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                    and i2[:3] == air1 
                                                    and int(i2[3:]) < h_departure]
    dep_lower_bound = max( R1_bound + R2_bound + R3_bound )

    # DEPARTURE UPPER BOUND
    #RULE 4 posterior Cargo or RULE 13 night flight
    R4_R13_bound = [dep_upper_cargo]
    #RULE 5 arrivals to the same origin airport STRICT INEQUALITY                                                      
    R5_bound = [ int(j2[3:])-1 for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                    and j2[:3] == air1 
                                                    and int(j2[3:]) > h_departure ]
    #RULE 6 departures from the same origin airport STRICT INEQUALITY
    R6_bound = [ int(i2[3:])-1 for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                    and i2[:3] == air1 
                                                    and int(i2[3:]) > h_departure]
    dep_upper_bound = min( R4_R13_bound + R5_bound + R6_bound )

    # ARRIVAL LOWER BOUND
    #RULE 7 previous Cargo or RULE 14 night flight
    R7_R14_bound = [arriv_lower_cargo]
    #RULE 8 arrivals to the same destination airport doesnt add a bound
    #RULE 9 departures from the same destination airport STRICT INEQUALITY
    R9_bound = [ int(i2[3:])+1 for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                    and i2[:3] == air2 
                                                    and int(i2[3:]) < h_arrival ]
    arriv_lower_bound = max( R7_R14_bound + R9_bound )

    # ARRIVAL UPPER BOUND
    #RULE 10 posterior Cargo
    R10_bound = [arriv_upper_cargo]
    #RULE 11 arrivals to the same destination airport doesnt add a bound
    #RULE 12 departures from the same destination airport
    R12_bound = [ int(i2[3:]) for i2,j2 in AF if sum(y[i2,j2,k] for k in K) == 1
                                                and i2[:3] == air2 
                                                and int(i2[3:]) >= h_arrival ]
    arriv_upper_bound = min( R10_bound + R12_bound )


    # vicinity
    Similar = [(i2,j2) for i2,j2 in AF if  sum(y[i2,j2,k] for k in K) == 0 
                                    and i2[:3] == air1 and j2[:3] == air2 
                                    and dep_lower_bound <= int(i2[3:]) <= dep_upper_bound 
                                    and arriv_lower_bound <= int(j2[3:]) <= arriv_upper_bound ]

    return Similar



def bounds_for_similar(Similar):
    min_flight = min(Similar, key=lambda x:int(x[0][3:]))
    max_flight = max(Similar, key=lambda x:int(x[1][3:]))
    a_hat = min_flight[0]
    b_hat = max_flight[1]
    return a_hat, b_hat



def generate_nflights_vicinity(y, A, AF, airports, K, vicinity_n):
    """
    for a given solution y, return a list of feasible solutions that replace at most vicinity_n flights for another one in its vecinities
    returns a FULL solution (for each arc in A, not only flight arcs in AF)
    """
    complete_vicinity = []  #[{neighbour1}, {neighbour2} ... ]
    vicinities = {}
    vicinities[0] = [y]
    # 1-flight vicinity for each flight previously computed
    for it in range(1,vicinity_n+1):
        vicinities[it] = []
        for other_y in vicinities[it-1]:
            for i_bar, j_bar in AF:                    
                for k_bar in K:
                    if other_y[i_bar, j_bar, k_bar] == 1:
                        Similar = generate_similar_flights(other_y, i_bar, j_bar, k_bar, AF, airports, K)
                        vicinity = generate_1flight_vicinity(other_y, i_bar, j_bar, k_bar, A, airports, K, Similar)
                        # add to the output
                        for other_y2 in vicinity:
                            # avoid duplicates
                            if other_y2 not in complete_vicinity:
                                complete_vicinity.append(other_y2)
                            vicinities[it].append(other_y2)

    return complete_vicinity



#############
# DEPRECATED #############################################################################################################
#############

def generate_1flight_vicinity(y, i_bar, j_bar, k_bar, A, airports, K, Similar):
    """
    for a given solution y, focusing on a specific ij flight operated by a k type aircraft (y[i,j,k]=1)
    return a list of feasible solutions that replace the observed flight for another one in its vecinity
    returns a FULL solution (for each arc in A, not only flight arcs in AF)
    """
    vicinity = []  #[{neighbour1}, {neighbour2}, ... ]

    for i2,j2 in Similar:  # vuelos similares al observado
        y_vecino = {}  # crear solucion nueva
        y_vecino[i2, j2, k_bar]  = 1  # vuelo similar es ahora operado por un avion del tipo observado
        y_vecino[i_bar, j_bar, k_bar]  = 0  # vuelo observado YA NO operado por un avion del tipo observado

        for i1,j1 in A:
            if (i1,j1) not in [ (i_bar, j_bar), (i2, j2) ]:  # arcos que no son el observado ni el similar que lo reemplazara
                for k in K:
                    y_vecino[i1, j1, k] = y[i1, j1, k]

        vicinity.append(y_vecino)

    return vicinity



def generate_2flights_vicinity(y, A, AF, airports, K, N=None, last_hour=None, n=None):
    """
    for a given solution y, return a list of feasible solutions that replace at most 2 flights for another one in its vecinities
    returns a FULL solution (for each arc in A, not only flight arcs in AF)
    """
    complete_vicinity = []  #[({neighbour1}, (i1,j1,k1), (i2,j2,k2)], ({neighbour2}, (i1,j1,k1), (i2,j2,k2) ... ]

    # 1-flight vicinity
    for i_bar1, j_bar1 in AF:                    
        for k_bar1 in K:
            if y[i_bar1, j_bar1, k_bar1] == 1:
                Similar1 = generate_similar_flights(y, i_bar1, j_bar1, k_bar1, AF, airports, K)
                vicinity1 = generate_1flight_vicinity(y, i_bar1, j_bar1, k_bar1, A, airports, K, Similar1)
                if last_hour != None:
                    input("modulo Improvements.py linea 313, esperando revision de imagenes")
                    print("i_bar1, j_bar1 = ", i_bar1, j_bar1)
                    print("vuelos similares: ", Similar1)
                    prt.print_test_2flight_vicinity_base(airports, N, A, AF, last_hour, K, n, y, i_bar1, j_bar1, Similar1)            
                # 1-flight vicinity for each flight in the 1-fv previously computed
                for other_y in vicinity1:
                    # add to the output
                    complete_vicinity.append( (other_y, (i_bar1,j_bar1,k_bar1), (None,None,None)) )
                    cont_segundas_vecindades = 0
                    for i_bar2, j_bar2 in AF:
                        # for each flight NOT in S(i_bar,j_bar) nor (i_bar,j_bar)
                        if (i_bar2,j_bar2) not in Similar1+[(i_bar1,j_bar1)]:               
                            for k_bar2 in K:
                                if other_y[i_bar2, j_bar2, k_bar2] == 1:
                                    # similar flights to the observed one
                                    Similar2 = generate_similar_flights(other_y, i_bar2, j_bar2, k_bar2, AF, airports, K)
                                    vicinity2 = generate_1flight_vicinity(other_y, i_bar2, j_bar2, k_bar2, A, airports, K, Similar2)    
                                    if last_hour != None:
                                        cont_segundas_vecindades += 1
                                        print(" i_bar2, j_bar2 = ", i_bar2, j_bar2)
                                        print(" vuelos similares: ", Similar2)
                                        prt.print_test_2flight_vicinity(airports, N, A, AF, last_hour, K, n, other_y, i_bar1, j_bar1, i_bar2,j_bar2, Similar1, Similar2, cont_segundas_vecindades)
                                    # add to the output
                                    for other_y2 in vicinity2:
                                        complete_vicinity.append( (other_y2, (i_bar1,j_bar1,k_bar1), (i_bar2,j_bar2,k_bar2)) )

    return complete_vicinity


if __name__ == "__main__":
    print("Module with functions for Order and T(y01, y02)")               

    airports = ["AAA", "BBB", "CCC"]
    days = 2
    K = [1]
    AF = []
    flight_duration = 3
    for air1 in airports:
        for air2 in airports:
            if air1 != air2:
                for h in range(0, days*24-flight_duration+1):
                    i = air1+str(h)
                    j = air2+str(h+flight_duration)
                    AF.append( (i,j) )

    y = { 
        ("AAA1", "BBB4", 1): 1,
        ("CCC3", "AAA6", 1): 1,
        ("AAA7", "CCC10", 1): 1,
        ("BBB9", "AAA12", 1): 1,
        ("AAA13", "CCC16", 1): 1,
        ("AAA14", "BBB17", 1): 1,
        ("CCC17", "AAA20", 1): 1,
        ("BBB20", "AAA23", 1): 1,
        # ("BBB12", "AAA15", 1): 1,  # vuelo extra que deberia amarrarlo
        # ("BBB22", "CCC25", 1): 1,  # vuelo extra nocturno interes
        ("BBB12", "CCC15", 1): 1  # vuelo de interes
        }
    
    for i,j in AF:
        for k in K:
            if (i,j,k) not in y.keys():
                y[i,j,k] = 0

    Similar = generate_similar_flights(y, "BBB12", "CCC15", 1, AF, airports, K)
    vicinity = generate_1flight_vicinity(y, "BBB12", "CCC15", 1, AF, airports, K, Similar)

    # Similar = generate_similar_flights(y, "BBB22", "CCC25", 1, AF, airports, K)
    # vicinity = generate_1flight_vicinity(y, "BBB22", "CCC25", 1, AF, airports, K, Similar)  # vecindad vuelo nocturno

    print(f"Hay {len(Similar)} vuelos similares")
    print(f"Hay {len(vicinity)} soluciones en la vecindad")

    for flight in Similar:
        print(flight)
