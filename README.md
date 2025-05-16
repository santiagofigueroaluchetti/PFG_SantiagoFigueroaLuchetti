# PFG_SantiagoFigueroaLuchetti
En este repositorio se adjunta el código referente a mi Proyecto de Fin de Grado así como las fuentes de datos públicas empleadas.
Los archivos que se adjuntan en el repositorio son los siguientes:


# 1. Limpieza_datosPOB.ipynb
Limpieza y transformacion de la base de datos de población, también une los datos de población a la geometría urbana de CDM.

Entra:
"conjunto_de_datos_ageb_urbana_09_cpv2020.csv"
'RPY_Manzanas_Sample.shp'
'RPY_Manzanas_Sample.dbf'
Sale:
"Reproyectada_datospublicos_SAMPLE_CDM.shp"


# 2. Limpieza_datosGASTO.ipynb
Limpieza y transformación de base de datos de tipología de gasto

Entra:
"MBI_MarketData_2023_MX_AGEB_Municipios.xlsx" 
Sale:
"Tipología_de_gasto_sample.xlsx"


# 2. Union_GASTO_GEOMETRIA.ipynb
Une los datos de tipología de gasto a la geometría urbana de CDM

Entra:
"Tipología_de_gasto_sample.xlsx"
Sale:
"Tipologia_de_gasto_enmanzana_SAMPLE.shp"


# 3. Union_POB_GASTO_PARADAS.ipynb
Une los datos públicos, de gasto y paradas de transporte público.

Entra:
"Tipologia_de_gasto_enmanzana_SAMPLE.shp"
"Reproyectada_datospublicos_SAMPLE_CDM.shp"
"manzanas_zmvm.shp"
Sale:
"Manzanas_PUBLICOS_TIPOLOGIA_PARADAS_SAMPLE2.shp"

# QGIS (proceso intermedio antes de entrar al codigo final):
Se aplican últimas transformaciones a la geometría de la prueba territorial de CDM con los datos ya finales agregados, los poblacionales, gasto y paradas.

Entra:
"Manzanas_PUBLICOS_TIPOLOGIA_PARADAS_SAMPLE2.shp"
Sale:
"BUENO_MAZANASPUBLICOSTIPOLOGIAPARADAS_SAMPLE.shp"


# 4. PFG_Codigo_completo.py
Codigo final donde se realiza el Proyecto de Fin de Grado de forma completa.

Se utiliza:
"RPY_AGEB_Sample.shp"
"RPY_AGEB_Sample.shx"
"BUENO_MAZANASPUBLICOSTIPOLOGIAPARADAS_SAMPLE.shp"
"PoisSampleLimpiosCDM.shp"
"cdmx_overturemaps.geoparquet"
