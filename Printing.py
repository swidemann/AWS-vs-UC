#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

import os
import matplotlib.pyplot as plt
import random as rd
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

rd.seed(1)


def _hanging_line(point1, point2):
    import numpy as np

    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b

    return (x,y)


def create_base(airports, nodes, archs, last_hour):
    plt.figure(figsize=(16, 12)) 
    plt.yticks([])
    inicio = 0
    fin = last_hour
    lineas = {}
    alturas = {}
    altura = 0
    for a in airports:
        alturas[a] = altura
        lineas[a] = plt.subplot()
        #lineas para aeropuertos
        lineas[a].plot((inicio, fin), (altura, altura), 'k', alpha=.5)
        lineas[a].text(inicio-0.6, altura, a, horizontalalignment='right', fontsize=14, backgroundcolor=(1., 1., 1., .3))
        altura += 1
    for key in airports:
        altura = alturas[key]
        for hora in range(0, fin + 1):
            if key+str(hora) in nodes:
                lineas[key].scatter(hora, altura, s=100, facecolor='w', edgecolor='k', zorder=9999)
    return lineas, alturas


def print_original_network(airports, nodes, archs, last_hour, instance, directory):
    directory = "{}/{}".format(directory, instance)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    # vuelos posibles
    for arco in archs:
        a_inicio = arco[0][:3]
        h_inicio = arco[0][3:]
        #aeropuerto y hora de llegada
        a_fin = arco[1][:3]
        h_fin = arco[1][3:]
        # linea
        lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), 'k', alpha=.5)
    plt.tight_layout()
    plt.savefig("{}/{}_original network.png".format(directory, instance))
    plt.close()


def print_aggregated_fs_aircraft_schedule(airports, nodes, archs, last_hour, k, n, y, instance, directory):
    directory = "{}/{}/FS".format(directory, instance)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in archs:
        cantidad = round(y[0][arco[0], arco[1], k].x)
        if cantidad > 0:
            text_col = "k"
            a_inicio = arco[0][:3]
            h_inicio = arco[0][3:]
            #aeropuerto y hora de llegada
            a_fin = arco[1][:3]
            h_fin = arco[1][3:]
            if a_inicio != a_fin:
                text_col = "b"
            # linea
            # point1 = ( int(h_inicio),  alturas[a_inicio])
            # point2 = ( int(h_fin),  alturas[a_fin])
            # x,y = _hanging_line(point1, point2)
            # lineas[a_inicio].plot(x, y, "b", alpha=.5)
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "b", alpha=.5)
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
    plt.tight_layout()
    plt.savefig("{}/{}_Aggregated_FS_schedule_type_{}.png".format(directory, instance, k))
    plt.close()


def print_aggregated_fs_TEST(airports, nodes, archs, last_hour, k, y0_param, name):

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in archs:
        cantidad = round(y0_param[arco[0], arco[1], k])
        if cantidad > 0:
            text_col = "k"
            a_inicio = arco[0][:3]
            h_inicio = arco[0][3:]
            #aeropuerto y hora de llegada
            a_fin = arco[1][:3]
            h_fin = arco[1][3:]
            if a_inicio != a_fin:
                text_col = "b"
            # linea
            # point1 = ( int(h_inicio),  alturas[a_inicio])
            # point2 = ( int(h_fin),  alturas[a_fin])
            # x,y = _hanging_line(point1, point2)
            # lineas[a_inicio].plot(x, y, "b", alpha=.5)
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "b", alpha=.5)
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


def print_test_1flight_vicinity(airports, nodes, archs, flight_archs, last_hour, K, n, y, i_bar, j_bar, vicinity, Similar, instance, directory):
    directory = "{}/{}/1flight_test".format(directory, instance)

    if not os.path.exists(directory):
        os.makedirs(directory)

    #base
    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in flight_archs:
        cantidad = round(sum( y[arco[0], arco[1], k] for k in K ))
        #aeropuerto y hora de salida
        a_inicio = arco[0][:3]
        h_inicio = arco[0][3:]
        #aeropuerto y hora de llegada
        a_fin = arco[1][:3]
        h_fin = arco[1][3:]
        if cantidad > 0:
            text_col = "k"
            if a_inicio != a_fin:
                text_col = "b"
            # linea
            line_col = "b"
            if arco[0] == i_bar and arco[1] == j_bar:
                line_col = "r"
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color=line_col, alpha=.5)
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
        # imprimir en gris vuelos con mismo OD que i_bar j_bar
        elif a_inicio == i_bar[:3] and a_fin == j_bar[:3]:
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color="k", alpha=.5)
    plt.tight_layout()
    plt.savefig("{}/{}_1flight_base.png".format(directory, instance))
    plt.close()

    #vicinity
    cont = 0
    for other_y in vicinity:
        cont += 1
        lineas, alturas = create_base(airports, nodes, archs, last_hour)
        for arco in flight_archs:
            cantidad = round(sum( other_y[arco[0], arco[1], k] for k in K ))
            if cantidad > 0:
                text_col = "k"
                a_inicio = arco[0][:3]
                h_inicio = arco[0][3:]
                #aeropuerto y hora de llegada
                a_fin = arco[1][:3]
                h_fin = arco[1][3:]
                if a_inicio != a_fin:
                    text_col = "b"
                # linea
                line_col = "b"
                if arco in Similar:
                    line_col = "r"
                lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color=line_col, alpha=.5)
                # cantidad transportada
                lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
        plt.tight_layout()
        plt.savefig("{}/{}_1flight_neighbour_{}.png".format(directory, instance, cont))
        plt.close()


def print_test_2flight_vicinity_base(airports, nodes, archs, flight_archs, last_hour, K, n, y, i_bar1, j_bar1, Similar1):
    directory = "2flight_test"

    if not os.path.exists(directory):
        os.makedirs(directory)

    #base
    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in flight_archs:
        cantidad = round(sum( y[arco[0], arco[1], k] for k in K ))
        #aeropuerto y hora de salida
        a_inicio = arco[0][:3]
        h_inicio = arco[0][3:]
        #aeropuerto y hora de llegada
        a_fin = arco[1][:3]
        h_fin = arco[1][3:]
        if cantidad > 0:
            text_col = "k"
            if a_inicio != a_fin:
                text_col = "b"
            # linea
            line_col = "b"
            if arco[0] == i_bar1 and arco[1] == j_bar1:
                line_col = "r"
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color=line_col, alpha=.5)
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
        # imprimir en naranjo vuelos similares a i_bar j_bar    
        elif (arco[0],arco[1]) in Similar1:
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color="brown", alpha=.5)    
    plt.tight_layout()
    plt.savefig("{}/2fv_firstvicinity.png".format(directory))
    plt.close()


def print_test_2flight_vicinity(airports, nodes, archs, flight_archs, last_hour, K, n, y, i_bar1, j_bar1, i_bar2,j_bar2, Similar1, Similar2, cont_segundas_vecindades):
    directory = "2flight_test"

    if not os.path.exists(directory):
        os.makedirs(directory)

    #base
    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in flight_archs:
        cantidad = round(sum( y[arco[0], arco[1], k] for k in K ))
        #aeropuerto y hora de salida
        a_inicio = arco[0][:3]
        h_inicio = arco[0][3:]
        #aeropuerto y hora de llegada
        a_fin = arco[1][:3]
        h_fin = arco[1][3:]
        if cantidad > 0:
            text_col = "k"
            if a_inicio != a_fin:
                text_col = "b"
            # linea
            line_col = "b"
            if (arco[0], arco[1]) in Similar1:  # de la 1-fv
                line_col = "brown"
            elif arco[0] == i_bar2 and arco[1] == j_bar2:  # observado i_bar2,j_bar2
                line_col = "darkgreen"   
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color=line_col, alpha=.5)
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), color=text_col, horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
        # imprimir en varde claro vuelos similares a i_bar j_bar    
        elif (arco[0],arco[1]) in Similar2:  # de la 2-fv
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), color="limegreen", alpha=.5)    
    plt.tight_layout()
    plt.savefig("{}/2fv_fixed_secondvicinity_{}.png".format(directory, cont_segundas_vecindades))
    plt.close()


def print_convergence(theta_values, Q_values, modelname, directory, instance, integer=False):
    directory = "{}/{}".format(directory, instance)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    iterations = range(1, len(theta_values)+ 1)
    plt.figure(figsize=(12,12))
    if integer:
        image_name = "Integer Convergence {}".format(modelname)
        plt.title("Convergence of Theta and Q(y), {}".format(modelname))
        Q_label = "Q(y tongo)"
    else:
        image_name = "Relaxed Convergence {}".format(modelname)
        plt.title("Convergence of Theta and Q_LP(y), {}".format(modelname))
        Q_label = "Q_LP(y tongo)"
    plt.xlabel("Iteration Number")
    plt.plot(iterations, theta_values, label="Theta tongo", color='b')
    plt.plot(iterations, Q_values, label=Q_label, color ='r')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("{}/{}.png".format(directory, image_name))
    plt.close()


def print_histograms(directory, size, S, Cargo, created_cargo_types, instance):
    if not os.path.exists(directory):
        os.makedirs(directory)

    cargo_types = set([v for v in created_cargo_types.values()])
    for s in S:
        for c_type in cargo_types:
            values = [round(size[s][item]) for item in Cargo if created_cargo_types[item] == c_type]
            image_name = "Histogram i-{} sizes Cargo type {} scenario {}".format(instance, c_type, s)
            bin_size = 100
            bins_number = int((max(values) - min(values))/bin_size)
            plt.hist(values, bins=bins_number)
            plt.xlabel("Kg")
            plt.ylabel("Number")
            plt.title(image_name)
            plt.tight_layout()
            plt.savefig("{}/{}.png".format(directory, image_name))
            plt.close()



##################
# Update Pending #############################################################################################################
##################

def print_fs_aircraft_schedule(airports, nodes, archs, last_hour, K, n, y, instance, directory):
    directory = "{}/{}/FS".format(directory, instance)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for k in K:
        for m in n[k]:
            for arco in archs:
                if y[0][k][arco[0], arco[1], m].x > 0.5:
                    a_inicio = arco[0][:3]
                    h_inicio = arco[0][3:]
                    #aeropuerto y hora de llegada
                    a_fin = arco[1][:3]
                    h_fin = arco[1][3:]
                    # linea
                    lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "b", alpha=.5)
    plt.tight_layout()
    plt.savefig("{}/{}_FS_schedule.png".format(directory, instance))
    plt.close()


def print_ss_aircraft_schedule(airports, nodes, archs, last_hour, K, n, y, scenario, instance, directory):
    directory = "{}/{}/SS/{}".format(directory, instance, scenario)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for k in K:
        for m in n[k]:
            for arco in archs:
                if y[scenario][k][arco[0], arco[1], m].x > 0.5:
                    a_inicio = arco[0][:3]
                    h_inicio = arco[0][3:]
                    #aeropuerto y hora de llegada
                    a_fin = arco[1][:3]
                    h_fin = arco[1][3:]
                    # linea
                    lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "r", alpha=.5)                  
    plt.tight_layout()
    plt.savefig("{}/{}_SS_schedule.png".format(directory, instance))
    plt.close()


def print_aggregated_ss_aircraft_schedule(airports, nodes, archs, last_hour, k, n, y, scenario, instance, directory):
    directory = "{}/{}/SS/{}".format(directory, instance, scenario)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in archs:
        cantidad = round(y[scenario][arco[0], arco[1], k].x)
        if cantidad > 0:
            a_inicio = arco[0][:3]
            h_inicio = arco[0][3:]
            #aeropuerto y hora de llegada
            a_fin = arco[1][:3]
            h_fin = arco[1][3:]
            # linea
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "r", alpha=.5)     
            # cantidad transportada
            lineas[a_inicio].text((int(h_inicio) + int(h_fin))/2, (float(alturas[a_inicio])+float(alturas[a_fin]))/2+0.05, int(cantidad), horizontalalignment='right', fontsize=10, backgroundcolor=(1., 1., 1., .3))
    plt.tight_layout()
    plt.savefig("{}/{}_Aggregated_SS_schedule_type_{}.png".format(directory, instance, k))
    plt.close()


def print_aggregated_aircraft_routing_specific_schedule(airports, nodes, archs, last_hour, k, m, y, scenario, instance, directory):
    if scenario == 0:
        directory = "{}/{}/FS/{}".format(directory, instance, scenario)
        image_name = "{}/{}_FS_k{}m{}_specific_schedule.png".format(directory, instance, k, m)
    else:
        directory = "{}/{}/SS/{}".format(directory, instance, scenario)
        image_name = "{}/{}_SS_k{}m{}_specific_schedule.png".format(directory, instance, k, m)

    if not os.path.exists(directory):
        os.makedirs(directory)

    lineas, alturas = create_base(airports, nodes, archs, last_hour)
    for arco in archs:         
        if y[scenario][k][arco[0], arco[1], m].x > 0.5:
            a_inicio = arco[0][:3]
            h_inicio = arco[0][3:]
            #aeropuerto y hora de llegada
            a_fin = arco[1][:3]
            h_fin = arco[1][3:]
            # linea
            lineas[a_inicio].plot((int(h_inicio), int(h_fin)), (alturas[a_inicio], alturas[a_fin]), "r", alpha=.5)                           
    plt.tight_layout()
    plt.savefig(image_name)
    plt.close()







if __name__ == "__main__":
    print("Modulo con funciones para imprimir redes de espacio-tiempo")