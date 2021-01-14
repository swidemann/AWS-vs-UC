# Models

## TEST UC vs AWS

## Software needed:
1. Python
2. gurobipy
3. xlrd (read parameters .xlsx)

## Modo de uso de los modelos
Para cada instancia d{#dias}s{#escenarios} a resolver, ejecutar:

    'python <modelo.py> <instance> air<# airports> <consolidate> <Lmethod> <hours> <NS or reset> <partial or leave empty>'
por ejemplo:

    'python ./FSC_sb_QTb_reactive_SingleSat.py d4s20 air4 ConsBinCont simpleL 3 partial'

excepto el modelo con cortes proactivos, en que hay que definir adicionalmente la vecindad:

    'python <modelo.py> <instance> air<# airports> <consolidate> <Lmethod> <hours> <vicinity_n> <NS or reset> <partial or leave empty>'
por ejemplo:

    'python ./FSC_sb_QTb_proactive_nfv_SingleSat.py d4s20 air4 ConsBinCont simpleL 3 1 partial'

1. modelo.py: archivo con el modelo a resolver
2. instance: instancia a resolver, formato d{#dias}s{#escenarios}
3. air #airports: cantidad de aeropuertos considerados, formato air{#aeropuertos}
4. consolidate: consolidación de carga, opciones son Default (default) o ConsBinCont (consolidar y despues relajar)
5. Lmethod: calculo de L, simpleL basta ya que el sizebounding domina cualquier calculo mas acotado
6. <hours: limite de tiempo para resolver el modelo
7. vicinity_n: solo para modelo con cortes proactivos, que vecindad considerar. Usar solo 1
8. NS or reset: crear desde 0 o reiniciar el satelite. Usar solo reset
9. partial or leave empty: resolucion parcial progresiva de satelites. Usar partial para obtener incumbentes rapido

## Parametros
En Generator.py, la linea 16

    'noshow = False  # False or True'
define si se considera noshow o no (noshow= True: existe noshow, noshow=False= no existe noshow)

la linea 18 

    'var_percentage = 50'
define el nivel de variacion entre un escenario y otro (var_percentage = 10, 30, 50)

## Q outsample
Cada modelo de descomposicion ("FSC_...") tiene en la line 65

    'compute_Q_outsample = True'
la variable que define si se requiere el calculo de Q outsample (que puede requerir ~8 horas por instancia resuelta)

## shell script
Los archivos .sh de ejemplo asumen que existe una carpeta donde ira a resolver, lo ideal es mantener una carpeta distinta para cada generador (show o noshow, nivel de variacion 10 30 o 50) 
