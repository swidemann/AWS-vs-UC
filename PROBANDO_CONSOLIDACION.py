#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:07 2019

@author: steffan
"""

import sys

instance = sys.argv[1]
noshow_str = sys.argv[2].upper()
if noshow_str not in ["NOSHOW", "SHOW"]:
    raise Exception
noshow = True if noshow_str == "NOSHOW" else False
var_percentage = int(sys.argv[3])
n_airports_str = sys.argv[4]
n_airports = int(n_airports_str[n_airports_str.find("air")+3:])



###############
### Default ###
###############

from Generator import generate_parameters, consolidate_cargo, generate_continous_parameters          

days, S, airports, Nint, Nf, Nl, N, AF, AG, A, K, nav, n, av, air_cap, air_vol, Cargo, created_cargo_types, OD, size, vol, ex, mand, cv, cf, ch, inc, lc, sc, delta, V, gap, tv, S_OUTSAMPLE, size_OUTSAMPLE, vol_OUTSAMPLE, inc_OUTSAMPLE, ex_OUTSAMPLE, mand_OUTSAMPLE, sc_OUTSAMPLE = generate_parameters(instance, n_airports, noshow, var_percentage)
Cargo_base, OD_base = Cargo, OD

def_Cargo, def_size, def_vol, def_inc, def_OD, def_mand, def_ex = Cargo, size, vol, inc, OD, mand, ex

############
### Cont ###
############
consolidated_cargo_BOOL, continuous_cargo_BOOL, consolidate_continuous_cargo, consolidate = False, True, False, "Cont"

## consolidate Cargo
#if consolidated_cargo_BOOL:
#    Cargo, size, vol, inc, OD, mand, ex  = consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, continuous_cargo=continuous_cargo_BOOL)

# create vol/size and inc/size for continuous Cargo
if continuous_cargo_BOOL:
    cont_incperkg, cont_volperkg = generate_continous_parameters(Cargo, size, vol, inc, S)


###############
### ConsBin ###
###############
consolidated_cargo_BOOL, continuous_cargo_BOOL, consolidate_continuous_cargo, consolidate = True, False, False, "ConsBin"

# consolidate Cargo
if consolidated_cargo_BOOL:
    consbin_Cargo, consbin_size, consbin_vol, consbin_inc, consbin_OD, consbin_mand, consbin_ex  = consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, consolidate_continuous_cargo=consolidate_continuous_cargo)

## create vol/size and inc/size for continuous Cargo
#if continuous_cargo_BOOL:
#    incperkg, volperkg = generate_continous_parameters(Cargo, size, vol, inc, S)


################
### ConsCont ###
################
consolidated_cargo_BOOL, continuous_cargo_BOOL, consolidate_continuous_cargo, consolidate = True, True, True, "ConsCont"

# consolidate Cargo
if consolidated_cargo_BOOL:
    conscont_Cargo, conscont_size, conscont_vol, conscont_inc, conscont_OD, conscont_mand, conscont_ex  = consolidate_cargo(Cargo, OD, size, vol, mand, ex, inc, air_cap, air_vol, n, K, S, consolidate_continuous_cargo=consolidate_continuous_cargo)

# create vol/size and inc/size for continuous Cargo
if continuous_cargo_BOOL:
    conscont_incperkg, conscont_volperkg = generate_continous_parameters(conscont_Cargo, conscont_size, conscont_vol, conscont_inc, S)






print("\n## COMPARANDO ##")
origins, destinations = [], []
# get all origins and destinations
for item in Cargo:
    o,d = OD[item]
    if o not in origins:
        origins.append(o)
    if d not in destinations:
        destinations.append(d)

for s in S:
    print(f"Scenario {s}")
    for o in origins:
        for d in destinations:
            for mand_in_this_s in [0, 1]:
                
                this_def_items = [item for item in def_Cargo if ( (o,d) == def_OD[item] and def_mand[s][item] == mand_in_this_s)]
                this_def_size = round(sum(size[s][item] for item in this_def_items), 2)
                this_def_vol = round(sum(vol[s][item] for item in this_def_items), 2)
                this_def_inc = round(sum(inc[s][item] for item in this_def_items), 2)

                this_cont_items = [item for item in Cargo if ((o,d) == OD[item] and mand[s][item] == mand_in_this_s)]
                this_cont_size = round(sum(size[s][item] for item in this_cont_items), 2)
                this_cont_vol = round(sum(cont_volperkg[s][item]*size[s][item] for item in this_cont_items), 2)
                this_cont_inc = round(sum(cont_incperkg[s][item]*size[s][item] for item in this_cont_items), 2)

                this_consbin_items = [item for item in consbin_Cargo if ((o,d) == consbin_OD[item] and consbin_mand[s][item] == mand_in_this_s)]
                this_consbin_size = round(sum(consbin_size[s][item] for item in this_consbin_items), 2)
                this_consbin_vol = round(sum(consbin_vol[s][item] for item in this_consbin_items), 2)
                this_consbin_inc = round(sum(consbin_inc[s][item] for item in this_consbin_items), 2)

                this_conscont_items = [item for item in conscont_Cargo if ((o,d) == conscont_OD[item] and conscont_mand[s][item] == mand_in_this_s)]
                this_conscont_size = round(sum(conscont_size[s][item] for item in this_conscont_items), 2)
                this_conscont_vol = round(sum(conscont_volperkg[s][item]*conscont_size[s][item] for item in this_conscont_items), 2)
                this_conscont_inc = round(sum(conscont_incperkg[s][item]*conscont_size[s][item] for item in this_conscont_items), 2)

                if len(this_def_items) + len(this_cont_items) + len(this_consbin_items) + len(this_conscont_items) > 0:
                    print(f"{o}->{d} mand: {mand_in_this_s}")
                    print(f" DEFAULT   size: {this_def_size}, vol: {this_def_vol}, inc: {this_def_inc}")                            
                    print(f" CONT      size: {this_cont_size}, vol: {this_cont_vol}, inc: {this_cont_inc}")
                    print(f" CONSBIN   size: {this_consbin_size}, vol: {this_consbin_vol}, inc: {this_consbin_inc}")
                    print(f" CONSCONT  size: {this_conscont_size}, vol: {this_conscont_vol}, inc: {this_conscont_inc}")


