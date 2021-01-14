# Models

## TEST UC vs AWS

## Software needed:
1. Python
2. gurobipy
3. xlrd (read parameters .xlsx)

## Modo de uso de los modelos
Para cada instancia d{#dias}s{#escenarios} a resolver, ejecutar:

    'python <modelo.py> <instance> <show or noshow> <var: 10,30,50> air<# airports> <consolidate> <Lmethod> <hours> <NS or reset> <satellite: partial or full> <Qout or noQout>'
por ejemplo:

    'python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 ConsBinCont simpleL 3 partial noQout'

excepto el modelo con cortes proactivos, en que hay que definir adicionalmente la vecindad:

    'python <modelo.py> <instance> <show or noshow> <var: 10,30,50> air<# airports> <consolidate> <Lmethod> <hours> <vicinity_n> <NS or reset> <satellite: partial or full> <Qout or noQout>'
por ejemplo:

    'python ./FSC_sb_QTb_reactive_SingleSat.py d4s${s} show 30 air4 ConsBinCont simpleL 3 1 partial noQout'

1. modelo.py: archivo con el modelo a resolver
2. instance: instancia a resolver, formato d{#dias}s{#escenarios}
3. show or noshow
4. var: variance level 10, 30 or 50
5. air #airports: cantidad de aeropuertos considerados, formato air{#aeropuertos}
6. consolidate: consolidación de carga, opciones son Default (default) o ConsBinCont (consolidar y despues relajar)
7. Lmethod: calculo de L, simpleL basta ya que el sizebounding domina cualquier calculo mas acotado
8. hours: limite de tiempo para resolver el modelo
9. vicinity_n: solo para modelo con cortes proactivos, que vecindad considerar. Usar solo 1
10. NS or reset: crear desde 0 o reiniciar el satelite. Usar solo reset
11. partial or full: resolucion parcial progresiva de satelites. Usar partial para obtener incumbentes rapido
12. Q outsample: Qout: solve, or noQout: dont solve

## shell script
Los archivos .sh de ejemplo asumen que existe una carpeta donde ira a resolver, lo ideal es mantener una carpeta distinta para cada generador (show o noshow, nivel de variacion 10 30 o 50) 
