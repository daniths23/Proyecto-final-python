### Clase 

#variable de texto
variable_frase = "Mi nombre es Daniela y estudio economía en la EPN"
print(variable_frase)

#lista de números
edad_eco2 =[21,22,23,24,25,22]
print(edad_eco2)

#diccionario

mis_materias = {"eco_2" : "L-M-V","eco_pub" : "J-V","formulacion" : "M","ing_finan" : "V","t_juegos" : "L-M"}
print(mis_materias)
############
vector_enteros=[14]*7
print(vector_enteros)

vector_flotantes=[9.8]*5
print(vector_flotantes)

diccionario = {"entero" : vector_enteros, "flotante" : vector_flotantes}
print(diccionario)

#CADENAS
cadena_simple = 'Hola a todos!'
cadena_doble=["Estudio economía","y estoy cursando sexto-séptimo semestre"]
print(cadena_doble)

#lectura de una tabla Excel usando pandas
import pandas as pd
imp_sri = pd.read_excel("ventas_SRI (1).xlsx")
print(imp_sri)