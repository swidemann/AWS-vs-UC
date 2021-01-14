#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=TShow
# Archivo de salida
#SBATCH --output=salida_v50.0TESTShowVar50Air4_QTb_r.txt
# Partición (Cola de trabajo)
#SBATCH --partition=full
# Reporte por correo
#SBATCH --mail-type=END
#SBATCH --mail-user=sowidemann@uc.cl
# Solicitud de cpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

cd /home1/sowidemann/v50.0TESTShowVar50Air4/
for s in {1,10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 ConsBinCont simpleL 3 partial noQout &> OUTPUT_v50.0ShowVar30Air4_ConsBinCont_QTb_r_SingleSat_partial_d4s${s}.txt;
done;

for s in {1,10,20,30}; do 
python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 Default simpleL 3 partial noQout &> OUTPUT_v50.0ShowVar30Air4_Default_QTb_r_SingleSat_partial_d4s${s}.txt;
done;
