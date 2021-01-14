#!/bin/bash

cd "/home/ubuntu/Experimentos/v49.1TESTShowVar50Air4"

for s in {10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} air4 ConsBinCont simpleL 3 partial &> OUTPUT_v49.1ShowVar30Air4_ConsBinCont_QTb_r_SingleSat_partial_d4s${s}.txt;
done;

for s in {10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} air4 Default simpleL 3 partial &> OUTPUT_v49.1ShowVar30Air4_Default_QTb_r_SingleSat_partial_d4s${s}.txt;
done;

wait
for f in *.txt; do dbxcli put $f v49.1ShowVar50/$f ; done
