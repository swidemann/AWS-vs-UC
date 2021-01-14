#!/bin/bash

cd "/home/ubuntu/Experimentos/v50.0TESTShowVar50Air4"

for s in {10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 ConsBinCont simpleL 3 partial noQout &> OUTPUT_v50.0ShowVar30Air4_ConsBinCont_QTb_r_SingleSat_partial_d4s${s}.txt;
done;

for s in {10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 Default simpleL 3 partial noQout &> OUTPUT_v50.0ShowVar30Air4_Default_QTb_r_SingleSat_partial_d4s${s}.txt;
done;

wait
for f in *.txt; do dbxcli put $f v50.0ShowVar50/$f ; done
