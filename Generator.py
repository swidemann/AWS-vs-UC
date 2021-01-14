#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

import xlrd
import numpy as np
import random as rd
import math

parameters_file = 'PARAMETERS.xlsx'

noshow = False
# from existing scenario 0, at most new size and vol in [original*(1-percentage/100), original*(1+percentage/100)] 
var_percentage = 50
# percentage of ODs that exists in every scenario
percentage_allways_exists = 0.3 if noshow else 1 #TODO
# probability of an OD to not show
p_noshow = 0.3 if noshow else 0 #TODO
# probability od an OD being mandatory
p_mandatory = 0.1
# cada cuanto se programa un vuelo base 6 DEFAULT
step = 6
# cantidad de escenarios outsample para probar Q real
nscenarios_OUTSAMPLE = 50  #TODO



def params_normal_to_lognormal(mu, sigma):
    log_mu = 2*np.log(mu) - 1/2*np.log(np.square(sigma) + np.square(mu))
    log_sigma = np.sqrt(-2*np.log(mu) + np.log(np.square(sigma) + np.square(mu)) )
    return log_mu, log_sigma

def time_between_nodes(node1, node2):  # SCL15 MIA25 returns 10
    return abs(int(node1[3:])-int(node2[3:]))

def generate_parameters(instance, n_airports=3, seed=1, seed_OUTSAMPLE=2):
    if seed != None:
        np.random.seed(seed)
        rd.seed(seed)

    ##############
    ### Inputs ###
    ##############

    # Airports
    master_airports = ["SCL", "GRU", "MIA", "VCP", "UIO", "ORD"]
    airports = master_airports[:n_airports]  # Santiago, Guarulhos, Miami

    # Fleet types
    # 1: Boeing 767-300F, 2: Airbus 300-600F
    K = [1]
    # number of available aircraft at the start by type and airport
    if n_airports == 3:
        nav = {('SCL0', 1): 3, ('GRU0', 1): 3, ('MIA0', 1): 3}
    elif n_airports == 4:
        nav = {('SCL0', 1): 2, ('GRU0', 1): 3, ('MIA0', 1): 2, ('VCP0', 1): 2}
    elif n_airports == 5:
        nav = {('SCL0', 1): 2, ('GRU0', 1): 2, ('MIA0', 1): 2, ('VCP0', 1): 2, ('UIO0', 1): 1}
    elif n_airports == 6:
        nav = {('SCL0', 1): 1, ('GRU0', 1): 2, ('MIA0', 1): 2, ('VCP0', 1): 2, ('UIO0', 1): 1, ('ORD0', 1): 1}

    # maximum variance allowed for number of aircrafts at the end at each airport
    delta = 0  # use 1 for small time horizons to allow more flights
    # for the epsilon-vicinity, flight legs that are somewhat near
    epsilon = 6  # hours
    # Cargo types
    cargo_tipes = [1, 2]  # technology, agriculture #TODO definir prob de un tipo u otro para cada aeropuerto
    # lower the number ODs tobe transported
    ODs_downscale = 1
    # Cargo available at midnight and must arrive before midnight
    ODs_in_order = True
    ############################
    ### Read PARAMETERS.xlsx ###
    ############################    

    # open parameters workbook
    wb = xlrd.open_workbook(parameters_file)

    # flight time with between airports with turn around
    flight_time = {}
    sheet = wb.sheet_by_name("Times")
    for ind1 in range(1, sheet.nrows):
        for ind2 in range(1, sheet.nrows):
            if ind1 != ind2:
                air1 = sheet.cell(ind1, 0).value                
                air2 = sheet.cell(0, ind2).value
                flight_time[(air1, air2)] = int(sheet.cell(ind1, ind2).value)

    # distances between airports
    distances = {}
    sheet = wb.sheet_by_name("Distances")
    for ind1 in range(1, sheet.nrows):
        for ind2 in range(1, sheet.nrows):
            if ind1 != ind2:
                air1 = sheet.cell(ind1, 0).value                
                air2 = sheet.cell(0, ind2).value
                distances[(air1, air2)] = int(sheet.cell(ind1, ind2).value)                

    # expected daily traffic in kgs 
    daily_traffic = {}
    sheet = wb.sheet_by_name("Daily_cargo")
    for ind1 in range(1, sheet.nrows):
        for ind2 in range(1, sheet.nrows):
            if ind1 != ind2:
                air1 = sheet.cell(ind1, 0).value                
                air2 = sheet.cell(0, ind2).value
                daily_traffic[(air1, air2)] = int(sheet.cell(ind1, ind2).value)

    # flight and holding costs per hour
    sheet = wb.sheet_by_name("Costs_per_hour")
    hourly_cf, hourly_cv, hourly_lc, hourly_sc = {}, {}, {}, {}
    for f_type in range(1, sheet.nrows):
        hourly_cf[f_type] = sheet.cell(f_type, 1).value  # USD/h
        hourly_cv[f_type] = sheet.cell(f_type, 2).value  # USD/(h*kg)
        hourly_lc[f_type] = sheet.cell(f_type, 3).value  # USD/h
        hourly_sc[f_type] = sheet.cell(f_type, 4).value  # USD/h
    hourly_ch = sheet.cell(1, 7).value  # USD/(h*kg)
    # print("hourly_cf, hourly_cv, hourly_lc, hourly_sc, hourly_ch ", hourly_cf, hourly_cv, hourly_lc, hourly_sc, hourly_ch)

    # demand simulation log-normal
    sheet = wb.sheet_by_name("Cargo_types")
    size_mu, size_sigma, vol_over_size_mu, vol_over_size_sigma = {}, {}, {}, {}
    for ind in range(1, sheet.nrows):
        size_mu[ind] = sheet.cell(ind, 1).value
        size_sigma[ind] = sheet.cell(ind, 2).value
        vol_over_size_mu[ind] = sheet.cell(ind, 3).value
        vol_over_size_sigma[ind] = sheet.cell(ind, 4).value
    # print("size_mu, size_sigma, vol_over_size_mu, vol_over_size_sigma ", size_mu, size_sigma, vol_over_size_mu, vol_over_size_sigma)

    # airport export Cargo type probabilities
    sheet = wb.sheet_by_name("Airports")
    cargo_type_prob = {}
    for ind in range(1, sheet.nrows):
        air = sheet.cell(ind, 0).value  
        prob = []              
        prob1 = sheet.cell(ind, 1).value
        prob2 = sheet.cell(ind, 2).value
        if prob1 == "" or prob2 == "":
            prob1, prob2 = 0.5, 0.5
        prob.append(prob1)
        prob.append(prob2)
        cargo_type_prob[air] = prob
    # print("cargo_type_prob ", cargo_type_prob)

    # read income per kg and distance, {cargo_type: cost}
    sheet = wb.sheet_by_name("Inc_per_km")
    minimum, normal, over_45, over_100, over_300, over_500, over_1000 = {}, {}, {}, {}, {}, {}, {}
    for c_type in range(1, sheet.nrows):
        # min cost per km
        minimum[c_type]  = sheet.cell(c_type, 1).value
        # cost per km and kg
        normal[c_type] = sheet.cell(c_type, 2).value
        over_45[c_type]  = sheet.cell(c_type, 3).value
        over_100[c_type]  = sheet.cell(c_type, 4).value
        over_300[c_type]  = sheet.cell(c_type, 5).value
        over_500[c_type]  = sheet.cell(c_type, 6).value
        over_1000[c_type]  = sheet.cell(c_type, 7).value    
    # print("minimum, normal, over_45, over_100, over_300, over_500, over_1000 ", minimum, normal, over_45, over_100, over_300, over_500, over_1000)    

    # fleet data
    sheet = wb.sheet_by_name("Fleet")
    air_cap, air_vol = {}, {}
    for f_type in range(1, sheet.nrows):
        air_cap[f_type] = sheet.cell(f_type, 2).value
        air_vol[f_type] = sheet.cell(f_type, 3).value
    # print("air_cap, air_vol ", air_cap, air_vol)


    #####################
    ### Read Instance ###
    #####################

    days = int(instance[1:instance.find("s")])
    scenarios = int(instance[instance.find("s")+1:])
    S = [i for i in range(1, scenarios + 1)]
    print("\n #### GENERATING PARAMETERS: Low Density Flight Network ####")
    print('DAYS: {}'.format(days))
    print('SCENARIOS: {}'.format(scenarios))
    print('NUMBER OF AIRPORTS: {}'.format(len(airports)))
    print(f"{percentage_allways_exists*100}% of Cargo allways exists")
    print(f"{p_noshow*100}% probability of noshow")
    print(f"{p_mandatory*100}% probability of mandatory")
    distribution = "Triangular"
    print(f"variation between scenarios: {distribution} multiplier in [{1 - float(var_percentage/100)}, {1 + float(var_percentage/100)}]")


    ###########################
    ### Generate Parameters ###
    ###########################

    # first and last nodes
    last_hour = 24*days
    Nf = []
    Nl = []
    airports_nodes = {}
    for airport in airports:
        Nf.append(airport + '0')
        Nl.append(airport + str(last_hour))
        airports_nodes[airport] = []  

    # flight arcs
    AF = []
    print(f"-- GENERADOR, VUELOS CADA {step} HORAS --")
    for aero in airports:
        for aero2 in airports:
            if aero != aero2:
                salida_base = 2
                while salida_base < last_hour:
                    if (aero, aero2) in flight_time.keys():
                        duration = flight_time[(aero, aero2)]
                    else:
                        duration = flight_time[(aero2, aero)]
                    salidas = [salida_base]
                    for salida in salidas:
                        llegada = salida + duration
                        if salida < last_hour and llegada < last_hour:
                            nodo_salida = aero + str(salida)
                            nodo_llegada = aero2 + str(llegada)
                            tupla = (nodo_salida, nodo_llegada)
                            AF.append(tupla)
                            if nodo_salida not in airports_nodes[aero]:
                                airports_nodes[aero].append(nodo_salida)
                            if nodo_llegada not in airports_nodes[aero2]:
                                airports_nodes[aero2].append(nodo_llegada)
                    salida_base += step


    # epsilon-vicinity
    V = {}
    gap = {}
    for arco1 in AF:
        v_arco = []
        for arco2 in AF:
            if arco1!=arco2 and arco1[0][:3]==arco2[0][:3] and arco1[1][:3]==arco2[1][:3]:
                # gap_value = abs(int(arco1[0][3:])-int(arco2[0][3:]))
                gap_value = time_between_nodes(arco1[0], arco2[0])  # time between take offs
                gap[arco1, arco2] = gap_value
                if gap_value <= epsilon:
                    v_arco.append(arco2)
        V[arco1] = v_arco         


    # Cargo  
    expected_cargo_kgs = {}
    Cargo = []
    OD = {}
    tiempo_minimo = max(flight_time.values()) + 1  # min time to deliver Cargo   
    item = 0
    for aero in airports:
        for aero2 in airports:
            if aero != aero2:
                generados = 0
                demanda_a_generar_kgs = daily_traffic[(aero, aero2)]*days
                expected_cargo_kgs[(aero, aero2)] = demanda_a_generar_kgs
                expected_od_size = sum(size_mu[i] for i in cargo_tipes)/len(cargo_tipes)  #TODO ajustar por porcentaje de cada aeropuerto
                demanda_a_generar = math.ceil(demanda_a_generar_kgs/expected_od_size*ODs_downscale)
                while generados < demanda_a_generar:
                    # od are only available from hour 1, and must reach before last hour -1
                    # -> are only relevant to Nint, no Nf or Nl
                    if ODs_in_order:
                        dia_salida = rd.randint(0, days-1)
                        h_salida = dia_salida*24   # available at 1AM
                        h_llegada = rd.randint(dia_salida+1, days)*24  # must arrive at 11PM
                    else:
                        h_salida = rd.randint(1, last_hour - tiempo_minimo - 1)
                        h_llegada = rd.randint(h_salida + tiempo_minimo, last_hour - 1)
                    od = (aero + str(h_salida), aero2 + str(h_llegada))
                    if h_salida >= 0 and h_llegada <= last_hour:
                        generados += 1
                        item += 1
                        Cargo.append(item)
                        OD[item] = od
    print('NUMBER OF CARGO: {}'.format(len(Cargo)))

    # subset of ODs that are present over every scenario
    allways_exist = rd.sample(Cargo, k=math.ceil(len(Cargo)*percentage_allways_exists))
    print('NUMBER OF CARGO FORCED TO ALLWAYS BE PRESENT: {} ({}% of total)'.format(len(allways_exist), percentage_allways_exists*100))

    # cargo exists
    ex = {}
    for s in S+[0]:
        ex_s = {}
        for item in Cargo:
            if item in allways_exist or np.random.choice([True, False], p=[1-p_noshow, p_noshow]):
                ex_s[item] = 1
            else:
                ex_s[item] = 0
        ex[s] = ex_s

    # transform normal parameters to log-normal
    log_size_mu, log_size_sigma, log_vol_over_size_mu, log_vol_over_size_sigma = {}, {}, {}, {}
    for i in cargo_tipes:
        l_s_m, l_s_s = params_normal_to_lognormal(size_mu[i], size_sigma[i])
        l_vos_m, l_vos_s =  params_normal_to_lognormal(vol_over_size_mu[i], vol_over_size_sigma[i])
        log_size_mu[i], log_size_sigma[i], log_vol_over_size_mu[i], log_vol_over_size_sigma[i] = l_s_m, l_s_s, l_vos_m, l_vos_s

    # Cargo dimentions
    size, vol = {}, {}
    created_cargo_types = {}  # {item: type}
    # create base scenario
    for s in [0]:
        size_s, vol_s = {}, {}
        for item in Cargo:
            item_origin = OD[item][0][:3]
            tipo = np.random.choice(cargo_tipes, p=cargo_type_prob[item_origin])
            created_cargo_types[item] = tipo
            size_s[item] = round(np.random.lognormal(mean = log_size_mu[tipo], sigma=log_size_sigma[tipo])*ex[s][item], 2)  # kg
            vol_over_size = np.random.lognormal(mean = log_vol_over_size_mu[tipo], sigma=log_vol_over_size_sigma[tipo])  # m3/kg
            vol_s[item] = round(vol_over_size*size_s[item]*ex[s][item], 2)  # m^3
        size[s] = size_s  
        vol[s] = vol_s
    # modify base scenario
    for s in S:
        size_s, vol_s = {}, {}
        for item in Cargo:
            if ex[0][item] == 0:  # created now, no reference dimentions
                item_origin = OD[item][0][:3]
                tipo = np.random.choice(cargo_tipes, p=cargo_type_prob[item_origin])
                size_s[item] = round(np.random.lognormal(mean = log_size_mu[tipo], sigma=log_size_sigma[tipo])*ex[s][item], 2)  # kg
                vol_over_size = np.random.lognormal(mean = log_vol_over_size_mu[tipo], sigma=log_vol_over_size_sigma[tipo])  # m3/kg
                vol_s[item] = round(vol_over_size*size_s[item]*ex[s][item], 2)  # m^3
            else:  # it already existed, now it can change on a given percentage
                variation = rd.triangular(-1*var_percentage,0 , var_percentage)
                size_s[item] = round(size[0][item]*(1 + float(variation/100))*ex[s][item], 2)
                vol_s[item] = round(vol[0][item]*(1 + float(variation/100))*ex[s][item], 2)
        size[s] = size_s  
        vol[s] = vol_s

    # mandatory
    mand = {}
    for s in [0]:  # base scenario
        mand_s = {}
        for item in Cargo:
            mandatory = 0
            if np.random.choice([True, False], p=[p_mandatory, 1-p_mandatory]) and ex[s][item] == 1:
                mandatory = 1
            mand_s[item] = mandatory
        mand[s] = mand_s
    for s in S:  # new emerging ods cant be mandatory
        mand_s = {}
        for item in Cargo:
            mandatory = 0
            if mand[0][item] == 1 and ex[s][item] == 1:  # if it was mandatory and still exists, then still is mandatory
                mandatory = 1
            mand_s[item] = mandatory
        mand[s] = mand_s   

    # add nodes for the origin and destination of Cargo
    for item in Cargo:
        od = OD[item]  # od = ([airport1][hour1], [airport2][hour2])
        for nodo in [od[0], od[1]]:
            aero = nodo[:3]
            hour = int(nodo[3:])
            if nodo not in airports_nodes[aero] and (hour != 0 and hour != last_hour):
                airports_nodes[aero].append(nodo)

    # intermediate nodes
    Nint = []
    for lista in airports_nodes.values():
        Nint += lista

    # ground arcs
    AG = []
    for aero in airports:
        h_salida = 0
        lista = airports_nodes[aero][:]
        lista.sort()
        while len(lista) != 0:
            h_llegada = last_hour
            for nodo in lista:
                hora = int(nodo[3:])
                if hora < h_llegada:
                    h_llegada = hora
            arco_terrestre = (
            aero + str(h_salida), aero + str(h_llegada))
            AG.append(arco_terrestre)
            if h_salida == 0 or aero + str(h_salida) not in lista:
                lista.remove(aero + str(h_llegada))
            h_salida = h_llegada

        arco_terrestre = (
        aero + str(h_salida), aero + str(last_hour))
        AG.append(arco_terrestre)

    A = AF + AG
    print('NUMBER OF AVAILABLE FLIGHT LEGS: {}'.format(len(AF)))
    print('NUMBER OF ARCS: {}'.format(len(A)))
    N = Nint + Nf + Nl
    print('NUMBER OF NODES: {}'.format(len(N)))
    aircraft_number = 0
    for aero in airports:
        nodo0 = aero + '0'
        for k in K:
            aircraft_number += nav[(nodo0, k)]

    print('NUMBER OF AIRCRAFT: {}'.format(aircraft_number))
    print('NUMBER OF AIRCRAFT TYPES: {}'.format(len(K)))
    n = {}
    for k in K:
        cont = 0
        for nodo0 in Nf:
            cont += nav[(nodo0, k)]
        n[k] = range(1, cont + 1)

    av = {}
    for k in K:
        id = 0
        for nodo0 in Nf:
            asignados = 0
            while asignados < nav[(nodo0, k)]:
                id += 1
                av[(nodo0, k, id)] = 1
                asignados += 1
                for otro_nodo in Nf:
                    if otro_nodo != nodo0:
                        av[(otro_nodo, k, id)] = 0

    # variable flight costs, fixed flight costs, holding costs
    cv = {}
    cf = {}
    ch = {}
    for k in K:
        for arco in AF:
            # arco_time = abs(int(arco[1][3:])-int(arco[0][3:]))
            arco_time = time_between_nodes(arco[1], arco[0])
            cv[(arco, k)] = round(hourly_cv[k]*arco_time, 2)
            cf[(arco, k)] = round(hourly_cf[k]*arco_time, 2)
    for arco in AG:
        # arco_time = abs(int(arco[1][3:])-int(arco[0][3:]))
        arco_time = time_between_nodes(arco[1], arco[0])
        ch[arco] = round(hourly_ch*arco_time, 2) 


    # income for Cargo
    inc = {}
    for s in S+[0]:
        inc[s] = {}
        for item in Cargo:
            if ex[s][item] == 0:
                inc[s][item] = 0
            else: 
                if size[s][item] < 45:
                    cost_per_kg_km = normal[created_cargo_types[item]]
                elif 45 <= size[s][item] < 100:
                    cost_per_kg_km = over_45[created_cargo_types[item]]
                elif 100 <= size[s][item] < 300:
                    cost_per_kg_km = over_100[created_cargo_types[item]]
                elif 300 <= size[s][item] < 500:
                    cost_per_kg_km = over_300[created_cargo_types[item]]
                elif 500 <= size[s][item] < 1000:
                    cost_per_kg_km = over_500[created_cargo_types[item]]
                elif 1000 <= size[s][item]:
                    cost_per_kg_km = over_1000[created_cargo_types[item]]
                cost_per_km = size[s][item]*cost_per_kg_km
                if cost_per_km < minimum[created_cargo_types[item]]:
                    cost_per_km = minimum[created_cargo_types[item]]
                air1, air2 = OD[item][0][:3], OD[item][1][:3]
                inc[s][item] = round(distances[(air1, air2)]*cost_per_km, 2)

    tv = round(hourly_lc[1]/100, 2)

    lc = {}  # long notice crew cost
    for k in K:
        for arco in AF:
            arco_time = time_between_nodes(arco[1], arco[0])
            lc[(arco, k)] = round(hourly_lc[k]*arco_time, 2)
    sc = {}  # short notice crew cost
    for s in S+[0]:  
        sc_s = {}
        for k in K:
            for arco in AF:
                arco_time = time_between_nodes(arco[1], arco[0])
                sc_s[(arco, k)] = round(hourly_sc[k]*arco_time, 2)
        sc[s] = sc_s


    print("***** PARAMETERS GENERATED *****\n")


    #################
    ### OUTSAMPLE ###
    #################

    
    if seed_OUTSAMPLE != None:
        np.random.seed(seed_OUTSAMPLE)
        rd.seed(seed_OUTSAMPLE)

    S_OUTSAMPLE = [i for i in range(1, nscenarios_OUTSAMPLE + 1)]
    size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE = {s: {} for s in S_OUTSAMPLE}, {s: {} for s in S_OUTSAMPLE}, {s: {} for s in S_OUTSAMPLE}, {s: {} for s in S_OUTSAMPLE}, {s: {} for s in S_OUTSAMPLE}, {s: {} for s in S_OUTSAMPLE}
    
    # modify base scenario
    for s in S_OUTSAMPLE:  
        #ex
        for item in Cargo:
            if item in allways_exist or np.random.choice([True, False], p=[1-p_noshow, p_noshow]):
                ex_OUTSAMPLE[s][item] = 1
            else:
                ex_OUTSAMPLE[s][item] = 0
        #size, vol
        for item in Cargo:
            if ex[0][item] == 0:  # created now, no reference dimentions
                item_origin = OD[item][0][:3]
                tipo = np.random.choice(cargo_tipes, p=cargo_type_prob[item_origin])
                size_OUTSAMPLE[s][item] = round(np.random.lognormal(mean = log_size_mu[tipo], sigma=log_size_sigma[tipo])*ex_OUTSAMPLE[s][item], 2)  # kg
                vol_over_size = np.random.lognormal(mean = log_vol_over_size_mu[tipo], sigma=log_vol_over_size_sigma[tipo])  # m3/kg
                vol_OUTSAMPLE[s][item] = round(vol_over_size*size_OUTSAMPLE[s][item]*ex_OUTSAMPLE[s][item], 2)  # m^3
            else:  # it already existed, now it can change on a given percentage
                variation = rd.triangular(-1*var_percentage,0 , var_percentage)
                size_OUTSAMPLE[s][item] = round(size[0][item]*(1 + float(variation/100))*ex_OUTSAMPLE[s][item], 2)
                vol_OUTSAMPLE[s][item] = round(vol[0][item]*(1 + float(variation/100))*ex_OUTSAMPLE[s][item], 2)
        #mand
        for item in Cargo:
            mandatory = 0
            if mand[0][item] == 1 and ex_OUTSAMPLE[s][item] == 1:  # if it was mandatory and still exists, then still is mandatory
                mandatory = 1
            mand_OUTSAMPLE[s][item] = mandatory
        #inc
        for item in Cargo:
            if ex_OUTSAMPLE[s][item] == 0:
                inc_OUTSAMPLE[s][item] = 0
            else: 
                if size_OUTSAMPLE[s][item] < 45:
                    cost_per_kg_km = normal[created_cargo_types[item]]
                elif 45 <= size_OUTSAMPLE[s][item] < 100:
                    cost_per_kg_km = over_45[created_cargo_types[item]]
                elif 100 <= size_OUTSAMPLE[s][item] < 300:
                    cost_per_kg_km = over_100[created_cargo_types[item]]
                elif 300 <= size_OUTSAMPLE[s][item] < 500:
                    cost_per_kg_km = over_300[created_cargo_types[item]]
                elif 500 <= size_OUTSAMPLE[s][item] < 1000:
                    cost_per_kg_km = over_500[created_cargo_types[item]]
                elif 1000 <= size_OUTSAMPLE[s][item]:
                    cost_per_kg_km = over_1000[created_cargo_types[item]]
                cost_per_km = size_OUTSAMPLE[s][item]*cost_per_kg_km
                if cost_per_km < minimum[created_cargo_types[item]]:
                    cost_per_km = minimum[created_cargo_types[item]]
                air1, air2 = OD[item][0][:3], OD[item][1][:3]
                inc_OUTSAMPLE[s][item] = round(distances[(air1, air2)]*cost_per_km, 2)
        #sc
        for (arco, k) in sc[0].keys():
                sc_OUTSAMPLE[s][(arco, k)] = sc[0][(arco, k)]

    return days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, created_cargo_types, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, S_OUTSAMPLE, size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE






def generate_average_scenario(S, OD, size, vol, inc, sc, mand, ex, AF):
    """
    for each od: over all scenarios compute
    average size, vol, inc, sc, ex, mand
    """
    av_size, av_vol, av_inc, av_ex, av_mand = {}, {}, {}, {}, {}
    for od in OD:
        this_size, this_vol, this_inc = 0, 0, 0
        for s in S:
            this_size += size[s][od]
            this_vol += vol[s][od]
            this_inc += inc[s][od]
        av_size[od] = round(this_size/len(S), 2)
        av_vol[od] = round(this_vol/len(S), 2)
        av_inc[od] = round(this_inc/len(S), 2)
    for od in OD:
        if av_size[od] > 0:
            av_ex[od] = 1
        else:
            av_ex[od] = 0
        av_mand[od] = mand[0][od]*av_ex[od]

    return av_size, av_vol, av_inc, av_ex, av_mand 





def generate_sizebound_parameters(Cargo, inc, size, vol, S):
    incperkg_sb, volperkg_sb = {}, {}
    for item in Cargo:
        incomes = [inc[s][item]/size[s][item] for s in S+[0] if size[s][item] > 0]
        if len(incomes) > 0:
            incperkg_sb[item] = round(max(incomes), 10)
        else:
            incperkg_sb[item] = 0
        volperkg_list = [vol[s][item]/size[s][item] for s in S+[0] if size[s][item] > 0]
        if len(volperkg_list) > 0:
            volperkg_sb[item] = round(min(volperkg_list), 10)  # menor vol/kg para facilitar restriccion de volumen
        else:
            volperkg_sb[item] = 0
    return incperkg_sb, volperkg_sb





def consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, consolidate_continuous_cargo=False):
    """
    Cargo consistent on each scenario
    for each scenario each Cargo item is a valid consolidation of the same original items
    """    
    test = False
    consolidated_Cargo, consolidated_OD = [], {}
    consolidated_size, consolidated_vol, consolidated_mand, consolidated_inc = {s: {} for s in S+[0]}, {s: {} for s in S+[0]}, {s: {} for s in S+[0]}, {s: {} for s in S+[0]}
    cargo_by_OD = {}
    # get all origins and destinations
    for item in Cargo:
        o,d = OD[item]
        if (o,d) not in cargo_by_OD.keys():
            cargo_by_OD[(o,d)] = []
        cargo_by_OD[(o,d)].append(item)

    # max size and vol for consolidated Cargo
    if consolidate_continuous_cargo:  # dont limit max size and vol for continuous formulation
        print("###### Consolidating Cargo: continuous Cargo, full consolidation ######")
    else:  # weighted average according to the fleet
        mult = 0.2
        total_aircrafts = sum( len(n[k]) for k in K)
        max_size = round(sum(air_cap[k]*len(n[k])/total_aircrafts for k in K)*mult, 2)
        max_vol = round(sum(air_vol[k]*len(n[k])/total_aircrafts for k in K)*mult, 2)
        print(f"###### Consolidating Cargo: binary Cargo, max size and vol {mult} of weigthed average cap: max size: {max_size}, max vol: {max_vol} ######")
    
    # consolidated is the same for each scenario
    cons_cargo_ID = 1
    for o,d in cargo_by_OD.keys():
        for mand_in_this_s in [0, 1]:
            # mandatory Cargo are mandatory in [0] and every scenario where it exists
            cargo_to_consolidate = [item for item in cargo_by_OD[(o,d)] if mand[0][item] == mand_in_this_s]
            if len(cargo_to_consolidate) > 0:
                # if test:
                #     print(f"Cargo to consolidate {o}{d}: {cargo_to_consolidate}")
                if consolidate_continuous_cargo:
                    consolidated_Cargo.append(cons_cargo_ID)
                    consolidated_OD[cons_cargo_ID] = (o, d)
                    for s in S+[0]:
                        consolidated_size[s][cons_cargo_ID] = round( sum(size[s][item] for item in cargo_to_consolidate), 2)
                        consolidated_vol[s][cons_cargo_ID] = round( sum(vol[s][item] for item in cargo_to_consolidate), 2)
                        consolidated_inc[s][cons_cargo_ID] = round( sum(inc[s][item] for item in cargo_to_consolidate), 2)
                        consolidated_mand[s][cons_cargo_ID] = mand_in_this_s if consolidated_size[s][cons_cargo_ID] > 0 else 0
                    #next consolidated
                    cons_cargo_ID += 1
                else:
                    this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}
                    while len(cargo_to_consolidate) > 0:
                        item_to_add = cargo_to_consolidate.pop()  #TODO mejorar 
                        can_be_added = True
                        special_case_unique_consolidated = False
                        # if the size and vol allows consolidation
                        for s in S+[0]:
                            if (this_cons_cargo_size[s] + size[s][item_to_add] <= max_size) and (this_cons_cargo_vol[s] + vol[s][item_to_add] <= max_vol):
                                pass
                            else:
                                can_be_added = False
                                # if some OD is bigger than the maximum, disregar the maximum and consolidate as a single item
                                if (size[s][item_to_add] >= max_size) or (vol[s][item_to_add] >= max_vol):
                                    special_case_unique_consolidated = True
                                    break
                        if can_be_added:  # cabe el sacado, agregar al consolidado
                            for s in S+[0]:
                                this_cons_cargo_size[s] += size[s][item_to_add]
                                this_cons_cargo_vol[s] += vol[s][item_to_add]
                                this_cons_cargo_inc[s] += inc[s][item_to_add]
                        else:  # no cabe el sacado, terminar el consolidado
                            consolidated_Cargo.append(cons_cargo_ID)
                            consolidated_OD[cons_cargo_ID] = (o, d)
                            for s in S+[0]:
                                consolidated_size[s][cons_cargo_ID] = round(this_cons_cargo_size[s], 2)
                                consolidated_vol[s][cons_cargo_ID] = round(this_cons_cargo_vol[s], 2)
                                consolidated_inc[s][cons_cargo_ID] = round(this_cons_cargo_inc[s], 2)
                                consolidated_mand[s][cons_cargo_ID] = mand_in_this_s if consolidated_size[s][cons_cargo_ID] > 0 else 0
                            #next consolidated
                            cons_cargo_ID += 1
                            this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}
                            if special_case_unique_consolidated:  #item is too big, create its own consolidated
                                consolidated_Cargo.append(cons_cargo_ID)
                                consolidated_OD[cons_cargo_ID] = (o, d)
                                for s in S+[0]:
                                    consolidated_size[s][cons_cargo_ID] = size[s][item_to_add]
                                    consolidated_vol[s][cons_cargo_ID] = vol[s][item_to_add]
                                    consolidated_inc[s][cons_cargo_ID] = inc[s][item_to_add]
                                    consolidated_mand[s][cons_cargo_ID] = mand_in_this_s if consolidated_size[s][cons_cargo_ID] > 0 else 0
                                #next consolidated
                                cons_cargo_ID += 1
                                this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}
                            else:  # item with standard size, append to pending to consolidate
                                cargo_to_consolidate.append(item_to_add)
                    # if there are no pending items to consolidate, finish the consolidated if there is something
                    if this_cons_cargo_size != {s: 0 for s in S+[0]}:
                        consolidated_Cargo.append(cons_cargo_ID)
                        consolidated_OD[cons_cargo_ID] = (o, d)
                        for s in S+[0]:
                            consolidated_size[s][cons_cargo_ID] = round(this_cons_cargo_size[s], 2)
                            consolidated_vol[s][cons_cargo_ID] = round(this_cons_cargo_vol[s], 2)
                            consolidated_inc[s][cons_cargo_ID] = round(this_cons_cargo_inc[s], 2)
                            consolidated_mand[s][cons_cargo_ID] = mand_in_this_s if consolidated_size[s][cons_cargo_ID] > 0 else 0
                        #next consolidated
                        cons_cargo_ID += 1
                        this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}, {s: 0 for s in S+[0]}
    
    # consolidated Cargo allways exists by construction
    consolidated_ex = {s: {} for s in S+[0]}
    for s in S+[0]:
        for item in consolidated_Cargo:
            if consolidated_size[s][item] > 0:
                consolidated_ex[s][item] = 1
            else:
                consolidated_ex[s][item] = 0

    if test:
        print("TESTING CONSOLIDATED")
    for s in S:
        if test:
            print(f"Scenario {s}")
        for o,d in cargo_by_OD.keys():
            for this_mand in [0,1]:
                items_originales = [item for item in Cargo if OD[item] == (o,d) and mand[s][item] == this_mand]
                orig_size = round( sum(size[s][item] for item in items_originales), 2)
                orig_vol = round( sum(vol[s][item] for item in items_originales), 2)
                orig_inc = round( sum(inc[s][item] for item in items_originales), 2)

                items_consolidados = [item for item in consolidated_Cargo if consolidated_OD[item] == (o,d) and consolidated_mand[s][item] == this_mand]
                consol_size = round( sum(consolidated_size[s][item] for item in items_consolidados), 2)
                consol_vol = round( sum(consolidated_vol[s][item] for item in items_consolidados), 2)
                consol_inc = round( sum(consolidated_inc[s][item] for item in items_consolidados), 2)

                if test:
                    if len(items_originales) + len(items_consolidados) > 0:
                        print(f"{o}->{d}, mand: {this_mand}")
                        print(f" ORIGINAL size: {orig_size}, vol: {orig_vol}, inc: {orig_inc}")
                        print(f" CONSOLID size: {consol_size}, vol: {consol_vol}, inc: {consol_inc}")
                if abs(orig_size-consol_size) > 2 or abs(orig_vol-consol_vol) > 2 or abs(orig_inc-consol_inc) > 2:
                    print(f"### ERROR AL CONSOLIDAR: scenario {s} {o}->{d} mand {this_mand}")
                    print(f" orig_size-consol_size = {orig_size-consol_size}, orig_vol-consol_vol = {orig_vol-consol_vol}, orig_inc-consol_inc = {orig_inc-consol_inc}")
    
    print(f"###### NUMBER OF CONSOLIDATED ODs: {len(consolidated_Cargo)} ######\n")
    return consolidated_Cargo, consolidated_size, consolidated_vol, consolidated_inc, consolidated_OD, consolidated_mand, consolidated_ex





def consolidate_cargo_average(Cargo, OD, size_av, vol_av, mand_av, ex_av, inc_av, air_cap, air_vol, n, K, consolidate_continuous_cargo=False):
    """
    Cargo consistent on each scenario
    for each scenario each Cargo item is a valid consolidation of the same original items
    """    
    test = False
    consolidated_Cargo, consolidated_OD = [], {}
    consolidated_size_av, consolidated_vol_av, consolidated_mand_av, consolidated_inc_av = {}, {}, {}, {}
    cargo_by_OD = {}
    # get all origins and destinations
    for item in Cargo:
        o,d = OD[item]
        if (o,d) not in cargo_by_OD.keys():
            cargo_by_OD[(o,d)] = []
        cargo_by_OD[(o,d)].append(item)

    # max size and vol for consolidated Cargo
    if consolidate_continuous_cargo:  # dont limit max size and vol for continuous formulation
        print("###### Consolidating Cargo: continuous Cargo, full consolidation ######")
    else:  # weighted average according to the fleet
        mult = 0.2
        total_aircrafts = sum( len(n[k]) for k in K)
        max_size = round(sum(air_cap[k]*len(n[k])/total_aircrafts for k in K)*mult, 2)
        max_vol = round(sum(air_vol[k]*len(n[k])/total_aircrafts for k in K)*mult, 2)
        print(f"###### Consolidating Cargo: binary Cargo, max size and vol {mult} of weigthed average cap: max size: {max_size}, max vol: {max_vol} ######")
    
    # consolidated is the same for each scenario
    cons_cargo_ID = 1
    for o,d in cargo_by_OD.keys():
        for mand_in_this_s in [0, 1]:
            # mandatory Cargo are mandatory in [0] and every scenario where it exists
            cargo_to_consolidate = [item for item in cargo_by_OD[(o,d)] if mand_av[item] == mand_in_this_s]
            if len(cargo_to_consolidate) > 0:
                # if test:
                #     print(f"Cargo to consolidate {o}{d}: {cargo_to_consolidate}")
                if consolidate_continuous_cargo:
                    consolidated_Cargo.append(cons_cargo_ID)
                    consolidated_OD[cons_cargo_ID] = (o, d)
                    consolidated_size_av[cons_cargo_ID] = round( sum(size_av[item] for item in cargo_to_consolidate), 2)
                    consolidated_vol_av[cons_cargo_ID] = round( sum(vol_av[item] for item in cargo_to_consolidate), 2)
                    consolidated_inc_av[cons_cargo_ID] = round( sum(inc_av[item] for item in cargo_to_consolidate), 2)
                    consolidated_mand_av[cons_cargo_ID] = mand_in_this_s if consolidated_size_av[cons_cargo_ID] > 0 else 0
                    #next consolidated
                    cons_cargo_ID += 1
                else:
                    this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = 0, 0, 0
                    while len(cargo_to_consolidate) > 0:
                        item_to_add = cargo_to_consolidate.pop()  #TODO mejorar 
                        can_be_added = True
                        special_case_unique_consolidated = False
                        # if the size and vol allows consolidation
                        if (this_cons_cargo_size + size_av[item_to_add] <= max_size) and (this_cons_cargo_vol + vol_av[item_to_add] <= max_vol):
                            pass
                        else:
                            can_be_added = False
                            # if some OD is bigger than the maximum, disregar the maximum and consolidate as a single item
                            if (size_av[item_to_add] >= max_size) or (vol_av[item_to_add] >= max_vol):
                                special_case_unique_consolidated = True
                        if can_be_added:  # cabe el sacado, agregar al consolidado
                            this_cons_cargo_size += size_av[item_to_add]
                            this_cons_cargo_vol += vol_av[item_to_add]
                            this_cons_cargo_inc += inc_av[item_to_add]
                        else:  # no cabe el sacado, terminar el consolidado
                            consolidated_Cargo.append(cons_cargo_ID)
                            consolidated_OD[cons_cargo_ID] = (o, d)
                            consolidated_size_av[cons_cargo_ID] = round(this_cons_cargo_size, 2)
                            consolidated_vol_av[cons_cargo_ID] = round(this_cons_cargo_vol, 2)
                            consolidated_inc_av[cons_cargo_ID] = round(this_cons_cargo_inc, 2)
                            consolidated_mand_av[cons_cargo_ID] = mand_in_this_s if consolidated_size_av[cons_cargo_ID] > 0 else 0
                            #next consolidated
                            cons_cargo_ID += 1
                            this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = 0, 0, 0
                            if special_case_unique_consolidated:  #item is too big, create its own consolidated
                                consolidated_Cargo.append(cons_cargo_ID)
                                consolidated_OD[cons_cargo_ID] = (o, d)
                                consolidated_size_av[cons_cargo_ID] = size_av[item_to_add]  #TODO deberia ser size_av[item_to_add]
                                consolidated_vol_av[cons_cargo_ID] = vol_av[item_to_add]
                                consolidated_inc_av[cons_cargo_ID] = inc_av[item_to_add]
                                consolidated_mand_av[cons_cargo_ID] = mand_in_this_s if consolidated_size_av[cons_cargo_ID] > 0 else 0
                                #next consolidated
                                cons_cargo_ID += 1
                                this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = 0, 0, 0
                            else:  # item with standard size, append to pending to consolidate
                                cargo_to_consolidate.append(item_to_add)
                    # if there are no pending items to consolidate, finish the consolidated if there is something
                    if this_cons_cargo_size != 0:
                        consolidated_Cargo.append(cons_cargo_ID)
                        consolidated_OD[cons_cargo_ID] = (o, d)
                        consolidated_size_av[cons_cargo_ID] = round(this_cons_cargo_size, 2)
                        consolidated_vol_av[cons_cargo_ID] = round(this_cons_cargo_vol, 2)
                        consolidated_inc_av[cons_cargo_ID] = round(this_cons_cargo_inc, 2)
                        consolidated_mand_av[cons_cargo_ID] = mand_in_this_s if consolidated_size_av[cons_cargo_ID] > 0 else 0
                        #next consolidated
                        cons_cargo_ID += 1
                        this_cons_cargo_size, this_cons_cargo_vol, this_cons_cargo_inc = 0, 0, 0
    
    # consolidated Cargo allways exists by construction
    consolidated_ex_av = {}
    for item in consolidated_Cargo:
        if consolidated_size_av[item] > 0:
            consolidated_ex_av[item] = 1
        else:
            consolidated_ex_av[item] = 0

    if test:
        print("TESTING CONSOLIDATED")
    for o,d in cargo_by_OD.keys():
        for this_mand in [0,1]:
            items_originales = [item for item in Cargo if OD[item] == (o,d) and mand_av[item] == this_mand]
            orig_size = round( sum(size_av[item] for item in items_originales), 2)
            orig_vol = round( sum(vol_av[item] for item in items_originales), 2)
            orig_inc = round( sum(inc_av[item] for item in items_originales), 2)

            items_consolidados = [item for item in consolidated_Cargo if consolidated_OD[item] == (o,d) and consolidated_mand_av[item] == this_mand]
            consol_size = round( sum(consolidated_size_av[item] for item in items_consolidados), 2)
            consol_vol = round( sum(consolidated_vol_av[item] for item in items_consolidados), 2)
            consol_inc = round( sum(consolidated_inc_av[item] for item in items_consolidados), 2)

            if test:
                if len(items_originales) + len(items_consolidados) > 0:
                    print(f"{o}->{d}, mand: {this_mand}")
                    print(f" ORIGINAL size: {orig_size}, vol: {orig_vol}, inc: {orig_inc}")
                    print(f" CONSOLID size: {consol_size}, vol: {consol_vol}, inc: {consol_inc}")
            if abs(orig_size-consol_size) > 2 or abs(orig_vol-consol_vol) > 2 or abs(orig_inc-consol_inc) > 2:
                print(f"### ERROR AL CONSOLIDAR: {o}->{d} mand {this_mand}")
                print(f" orig_size-consol_size = {orig_size-consol_size}, orig_vol-consol_vol = {orig_vol-consol_vol}, orig_inc-consol_inc = {orig_inc-consol_inc}")
    
    print(f"###### NUMBER OF CONSOLIDATED ODs: {len(consolidated_Cargo)} ######\n")
    return consolidated_Cargo, consolidated_size_av, consolidated_vol_av, consolidated_inc_av, consolidated_OD, consolidated_mand_av, consolidated_ex_av



def generate_continous_parameters(Cargo, size, vol, inc, S):
    incperkg = {s: {} for s in S+[0]}
    volperkg = {s: {} for s in S+[0]}
    for s in S:
        for item in Cargo:
            if size[s][item] > 0:
                incperkg[s][item] = round(inc[s][item]/size[s][item], 10)
                volperkg[s][item] = round(vol[s][item]/size[s][item], 10)
            else:
                incperkg[s][item] = 0
                volperkg[s][item] = 0

    return incperkg, volperkg

def generate_continous_parameters_average(Cargo, size_av, vol_av, inc_av):
    incperkg = {}
    volperkg = {}
    for item in Cargo:
        if size_av[item] > 0:
            incperkg[item] = round(inc_av[item]/size_av[item], 10)
            volperkg[item] = round(vol_av[item]/size_av[item], 10)
        else:
            incperkg[item] = 0
            volperkg[item] = 0

    return incperkg, volperkg



if __name__ == '__main__':
    print('Module to create instances')
