import geopandas as gpd
import pandas as pd
import folium
import h3
from shapely.geometry import Polygon, MultiPolygon
import fiona
import ast
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import branca.colormap as cm





with fiona.Env(SHAPE_RESTORE_SHX="YES"):
    gdf1 = gpd.read_file('RPY_AGEB_Sample.shp') #el ar que es como mazo grande
    gdf2 = gpd.read_file('RPY_AGEB_Sample.shx') #el a que es como ageb
    gdf3 = gpd.read_file('BUENO_MAZANASPUBLICOSTIPOLOGIAPARADAS_SAMPLE.shp')
                        
#Confirmar y asignar EPSG:4326 si no está definido
if gdf1.crs is None or gdf1.crs.to_string() != 'EPSG:4326':
    gdf1.set_crs('EPSG:4326', inplace=True, allow_override=True)

if gdf2.crs is None or gdf2.crs.to_string() != 'EPSG:4326':
    gdf2.set_crs('EPSG:4326', inplace=True, allow_override=True)

if gdf3.crs is None or gdf3.crs.to_string() != 'EPSG:4326':
    gdf3.set_crs('EPSG:4326', inplace=True, allow_override=True)

    #APLICANDO LA TRANSOFORMACION DE LOS DATOS DE GASTO
import pandas as pd
ids = pd.read_csv("ids_mza.csv", encoding="ISO-8859-1")
ids = ids.rename(columns={'folio_mza': 'CODE'})
gdf3 = pd.merge(gdf3, ids, on='CODE', how='inner')

dict_grupos_csp_ids = {
    "CSP01_EURO": ["ids_cassi"],
    "CSP02_EURO": ["ids_cbdj"],
    "CSP03_EURO": ["ids_cbdj", "ids_ctelj"],
    "CSP04_EURO": ["ids_ccevj"],
    "CSP05_EURO": ["ids_cbdj"],
    "CSP06_EURO": ["ids_ccevj"],
    "CSP07_EURO": ["ids_ccevj"],
    "CSP08_EURO": ["ids_ccevj"],
    "CSP09_EURO": ["ids_casi"],
    "CSP10_EURO": ["ids_ctelj"],
    "CSP11_EURO": ["ids_ctelj"],
    "CSP12_EURO": ["ids_cbdj", "ids_ctelj"],
    "CSP13_EURO": ["ids_rei"],
    "CSP14_EURO": ["ids_cbdj", "ids_ctelj"],
    "CSP15_EURO": ["ids_cbdj"],
    "CSP16_EURO": ["ids_cassi"],
    "CSP17_EURO": ["ids_cassi", "ids_casi"],
    "CSP18_EURO": ["ids_ccevj"],
    "CSP19_EURO": ["ids_cbdj", "ids_ctelj"],
    "CSP20_EURO": ["ids_cbdj", "ids_ctelj"]
}
def estimar_csp_con_ids(df, dict_grupos_csp_ids, normalizar=True):
    """
    Crea columnas nuevas CSPxx_EURO_estimado multiplicando por un score de IDS relacionado.

    Args:
        df: GeoDataFrame con columnas CSP e IDS.
        dict_grupos_csp_ids: diccionario {CSP: [IDS1, IDS2, ...]}.
        normalizar: si True, normaliza el score IDS entre 0 y 1.

    Returns:
        GeoDataFrame con columnas CSPxx_EURO_estimado.
    """
    df = df.copy()

    for csp_var, ids_list in dict_grupos_csp_ids.items():
        score_col = f"{csp_var}_score_ids"

        # Verificación de columnas
        columnas_necesarias = [csp_var] + ids_list
        if not all(col in df.columns for col in columnas_necesarias):
            print(f"❌ Columnas faltantes para {csp_var}: {columnas_necesarias}")
            continue

        # Asegurarse de que las columnas sean numéricas
        try:
            df[csp_var] = pd.to_numeric(df[csp_var], errors='coerce')
            for var in ids_list:
                df[var] = pd.to_numeric(df[var], errors='coerce')
        except Exception as e:
            print(f"❌ Error convirtiendo a numérico para {csp_var}: {e}")
            continue

        # 1. Calcular score IDS (suma de variables relacionadas)
        df[score_col] = df[ids_list].sum(axis=1)

        # 2. Normalizar si se pide
        if normalizar:
            min_val = df[score_col].min()
            max_val = df[score_col].max()
            df[score_col] = (df[score_col] - min_val) / (max_val - min_val + 1e-9)

        # 3. Multiplicar para obtener el CSP estimado
        estimado_col = f"{csp_var}_estimado"
        df[estimado_col] = df[csp_var] * df[score_col]


    return df
gdf3 = estimar_csp_con_ids(gdf3, dict_grupos_csp_ids)

# Generar los nombres de las columnas a eliminar
columnas_a_eliminar = [f'CSP{i:02}_EURO' for i in range(1, 21)]

# Eliminar las columnas del DataFrame
gdf3 = gdf3.drop(columns=columnas_a_eliminar, errors='ignore')

# Lista de columnas que deseas eliminar
columnas_a_eliminar = [
    'pob', 'pob_nbi', 'ids_ccevj', 'ids_csj', 'ids_caej', 'ids_ctelj', 'ids_cbdj',
    'ids_rei', 'ids_cassi', 'ids_casi', 'idsm', 'e_idsm',
    'CSP01_EURO_score_ids', 'CSP02_EURO_score_ids', 'CSP03_EURO_score_ids',
    'CSP04_EURO_score_ids', 'CSP05_EURO_score_ids', 'CSP06_EURO_score_ids',
    'CSP07_EURO_score_ids', 'CSP08_EURO_score_ids', 'CSP09_EURO_score_ids',
    'CSP10_EURO_score_ids', 'CSP11_EURO_score_ids', 'CSP12_EURO_score_ids',
    'CSP13_EURO_score_ids', 'CSP14_EURO_score_ids', 'CSP15_EURO_score_ids',
    'CSP16_EURO_score_ids', 'CSP17_EURO_score_ids', 'CSP18_EURO_score_ids',
    'CSP19_EURO_score_ids', 'CSP20_EURO_score_ids'
]

# Filtrar para eliminar solo las columnas que NO terminan en '_estimado'
columnas_a_eliminar = [col for col in columnas_a_eliminar if not col.endswith('_estimado')]

# Eliminar las columnas del DataFrame
gdf3 = gdf3.drop(columns=columnas_a_eliminar, errors='ignore')

# Renombrar las columnas que terminan en '_estimado'
gdf3.rename(
    columns={col: col.replace('_estimado', '') for col in gdf3.columns if col.endswith('_estimado')},
    inplace=True
)

# Crear la malla H3 con índices H3 como columna 'ID'
def generate_h3_grid_with_ids(gdf, resolution=10):
    h3_indices = set()

    for geom in gdf.geometry:
        if geom.is_valid:
            if geom.geom_type == 'Polygon':
                geoms = [geom]
            elif geom.geom_type == 'MultiPolygon':
                geoms = list(geom.geoms)
            else:
                continue

            for poly in geoms:
                hexes = h3.polyfill_geojson({'type': 'Polygon', 'coordinates': [list(poly.exterior.coords)]}, resolution)
                h3_indices.update(hexes)

    h3_geometries = []
    h3_ids = []

    for h3_index in h3_indices:
        h3_geometries.append(Polygon(h3.h3_to_geo_boundary(h3_index, geo_json=True)))
        h3_ids.append(h3_index)  # Usamos el índice real como ID

    return gpd.GeoDataFrame({'ID': h3_ids, 'geometry': h3_geometries}, crs='EPSG:4326')

# Crear la malla con IDs reales H3
h3_gdf = generate_h3_grid_with_ids(gdf2, resolution=10)

gdf3 = gdf3.drop_duplicates(subset='CODE_2', keep='first')
gdf3 = gdf3.drop_duplicates(subset='CODE', keep='first')

# Lista de IDs a eliminar
ids_a_eliminar = ['0901000010135073', '0901000010135072', '0901000010135074','0901000010135071', '0901000010135075', '0901000010135070', '0901000010135076','0901000010135075', '0901000010135069', '0901000010135077', '0901000010135068','0901000010135031', '0901000010135050', '0901000010135051', '0901000010135052','0901000010135053', '0901000010135054', '0901000010135058', '0901000010135059','0901000010135057', '0901000010135060', '0901000010135061', '0901000010135065','0901000010135066', '0901000010135067', '0901000010135064', '0901000010135062','0901000010135063']  # reemplaza con los que necesites

# Filtrar el DataFrame excluyendo esos IDs
gdf3 = gdf3[~gdf3["CODE"].isin(ids_a_eliminar)]

if gdf3.crs is None or gdf3.crs.to_string() != 'EPSG:4326':
    gdf3.set_crs('EPSG:4326', inplace=True, allow_override=True)
if h3_gdf.crs is None or gdf3.crs.to_string() != 'EPSG:4326':
    h3_gdf.set_crs('EPSG:4326', inplace=True, allow_override=True)

    
gdf = gpd.overlay( h3_gdf, gdf3, how='identity')
# Calcular el área
# Convertir a un CRS en metros para cálculos precisos del área (EPSG:3857 es una opción común)
gdf = gdf.to_crs("EPSG:3857")
gdf['area'] = gdf.geometry.area
# Filtrar eliminando las filas donde el área es 0 o NaN
gdf_filtered = gdf[(gdf['area'] > 0) & (gdf['area']!= '*')]
#gdf_filtered = gdf[(gdf['ID'] == 3805)]
gdf_filtered.head(10)

gdf3 =gdf3.to_crs("EPSG:3857")
gdf3['area_total'] = gdf3.geometry.area
# Filtrar eliminando las filas donde el área es 0 o NaN
gdf3_filtered = gdf3[(gdf3['area_total'] > 0) & (gdf3['area_total']!= '*')]


import numpy as np
merged_gdf = gdf.merge(gdf3[['CODE', 'area_total']], on='CODE', how='left')
merged_gdf['area_percentage'] = (merged_gdf['area'] / merged_gdf['area_total']) * 100
merged_gdf = merged_gdf.replace('*', 0)

print(merged_gdf.columns.tolist())

columnas_a_eliminar = ['CTRYCODE','CTRYCODE_2','CTRYCODE_x', 'MICROCODE_', 'NOM_ENT', 'NAME_x', 'NAME_y', 'CTRYCODE_y', 'MICROCOD_1','VPH_SINTIC']  # Reemplaza con las que quieres eliminar
merged_gdf = merged_gdf.drop(columns=columnas_a_eliminar, errors='ignore')

# Definir las columnas a excluir SOLO DEJO LAS POBLACIONALES QUE SON LAS PORCENTUALES ES DECIR QUITO TODO LO QUE NO ES POBLACIONAL
exclude_columns = ['','','CODE_2','NAME','MICROCODE','NAME_2','CVEGEO','NAME_y','MICROCOD_1','CTRYCODE_y','ID', 'geometry', 'area', 'area_total', 'area_percentage', 'CODE', 'NOM_MUN','P_T_TG','P_PRM_TG','PP_EURO_TG','CSP01_EURO','CSP02_EURO','CSP03_EURO','CSP04_EURO','CSP05_EURO','CSP06_EURO','CSP07_EURO','CSP08_EURO','CSP09_EURO','CSP10_EURO','CSP11_EURO','CSP12_EURO','CSP13_EURO','CSP14_EURO','CSP15_EURO','CSP16_EURO','CSP17_EURO','CSP18_EURO','CSP19_EURO','CSP20_EURO','CTRYCODE_x','MICROCODE_','NAME_x','NOM_ENT','Metro', 'Suburbano', 'Metrobus', 'Tren_Liger', 'Trolebus', 'RTP', 'Trole_elev', 'T_Concesio', 'Ecobici', 'Cablebus', 'Cobertura','OID_1','POB1']

# Excluir las columnas de `merged_gdf_example` y seleccionar las columnas numéricas restantes
numeric_cols = merged_gdf.drop(columns=exclude_columns, errors='ignore').replace('N/D', np.nan)
# Convertir todas las columnas de `numeric_cols` a tipo float
numeric_cols = numeric_cols.astype(float)
#print(numeric_cols)

# Asegurarnos de que las columnas en `numeric_cols.columns` dentro de `merged_gdf` sean de tipo float
merged_gdf[numeric_cols.columns] = merged_gdf[numeric_cols.columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

#merged_gdf['area_percentage'] = pd.to_numeric(merged_gdf['area_percentage'], errors='coerce').fillna(0).astype(float)
merged_gdf['area_percentage'] = merged_gdf['area_percentage'].fillna(0)

# Realizar la operación multiplicando columnas numéricas por 'area_percentage' y dividiendo entre 100
merged_gdf[numeric_cols.columns] = merged_gdf[numeric_cols.columns].multiply(merged_gdf['area_percentage'], axis=0) / 100

merged_gdf =merged_gdf.to_crs("EPSG:4326")
gdf3 =gdf3.to_crs("EPSG:4326")

from shapely.ops import unary_union

# 🔥 **PASO 1: Identificar columnas de tipología de gasto**
gasto_columns = [col for col in merged_gdf.columns if col.endswith('_TG') or col.endswith('_EURO')]
paradas = ['Metro','Suburbano','Metrobus','Tren_Liger','Trolebus','RTP','Trole_elev','T_Concesio','Ecobici','Cablebus','Cobertura']
nombres = ['NAME']

# Suponiendo que ya tienes el DataFrame cargado en `df`
# Eliminar las columnas no deseadas
exclude_columns2 = ['CODE', 'NOM_MUN', 'area', 'area_total', 'area_percentage']
merged_gdf = merged_gdf.drop(columns=exclude_columns2, errors ='ignore')
# Agrupar por ID, sumando todas las columnas numéricas y manteniendo la primera geometría


df_agrupado = merged_gdf.groupby('ID', as_index=False).agg({
    **{col: 'sum' for col in merged_gdf.columns if col not in ['ID', 'geometry','MICROCODE','NAME'] + gasto_columns + paradas},  # Sumar variables numéricas
    'geometry': lambda x: unary_union(x),  # Unir todas las geometrías en un solo polígono
    **{col: 'first' for col in gasto_columns + paradas + nombres}  # Mantener la primera ocurrencia para tipología de gasto y paradas
})

# Convertir df_agrupado en un GeoDataFrame, usando la columna 'geometry'
df_agrupado = gpd.GeoDataFrame(df_agrupado, geometry='geometry')
df_agrupado.set_crs("EPSG:4326", inplace=True)

#CARGA DE LOS POIS
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 🔥 **PASO 1: Cargar el shapefile de los POIs**
pois_gdf = gpd.read_file("PoisSampleLimpiosCDM.shp")

# 🔥 **PASO 2: Asegurar que los POIs y la malla H3 tienen el mismo CRS**
pois_gdf = pois_gdf.to_crs("EPSG:4326")
df_agrupado = df_agrupado.to_crs("EPSG:4326")

# 🔥 **PASO 3: Hacer el 'spatial join' entre POIs y la malla H3**
pois_joined = gpd.sjoin(pois_gdf, df_agrupado[['ID', 'geometry']], how="inner", predicate="within")

# 🔥 **PASO 4: Contar cuántos POIs hay por celda de H3**
pois_count = pois_joined.groupby('ID').size().reset_index(name='num_POIs')

# 🔥 **PASO 5: Agrupar los datos de los POIs en una lista dentro de cada celda H3**
pois_data = pois_joined.groupby("ID").apply(lambda x: x.to_dict(orient="records")).reset_index(name="pois_data")

# 🔥 **PASO 6: Fusionar la información de los POIs en `df_agrupado`**
df_agrupado = df_agrupado.merge(pois_count, on="ID", how="left").fillna({'num_POIs': 0})
df_agrupado = df_agrupado.merge(pois_data, on="ID", how="left")

# ✅ **Resultado Final**

df_agrupado



# 🔧 Lista de claves que quieres eliminar de cada POI
claves_a_eliminar = ["geometry", "fsq_place_", "fsq_catego", "latitude", "longitude", "index_right", "MICROCODE", "ID", "CTRYCODE", "NAME_2", "name"]

def limpiar_pois(pois):
    if isinstance(pois, float) and np.isnan(pois):
        return []
    try:
        if isinstance(pois, str):
            pois = ast.literal_eval(pois)
        if not isinstance(pois, list):
            return []
        return [
            {k: v for k, v in poi.items() if k not in claves_a_eliminar}
            for poi in pois if isinstance(poi, dict)
        ]
    except Exception:
        return []

# Aplicar la limpieza
df_agrupado["pois_data"] = df_agrupado["pois_data"].apply(limpiar_pois)

# Cargar el archivo GeoParquet
gdf_roads2 = gpd.read_file("cdmx_overturemapslimpio_sample.shp")
gdf_roads2

# 🚀 1. Asegurar que CRS es correcto
gdf_roads2 = gdf_roads2.to_crs(df_agrupado.crs)

# 🚀 2. Realizar el Spatial Join para asignar carreteras a celdas H3
roads_in_h3 = gpd.sjoin(gdf_roads2, df_agrupado[['ID', 'geometry']], how="inner", predicate="intersects")

# 🚀 3. Crear variables de conteo
# Contar cuántas carreteras y rieles hay en cada celda H3
roads_in_h3["num_road_OMR"] = (roads_in_h3["subtype"] == "road").astype(int)
roads_in_h3["num_rail_OMR"] = (roads_in_h3["subtype"] == "rail").astype(int)

# Crear variables de conteo para cada tipo de `class`
road_types = ['residential', 'path', 'track', 'footway', 'secondary', 'tertiary', 
              'cycleway', 'service', 'steps', 'unclassified', 'primary', 'motorway', 
              'living_street', 'trunk', 'pedestrian', 'unknown']

for road_type in road_types:
    roads_in_h3[f"num_{road_type}_OMR"] = (roads_in_h3["class"] == road_type).astype(int)

# 🚀 4. Agrupar por ID de celda H3 **EXCLUYENDO ID Y GEOMETRÍA**
gdf_roads_summary = roads_in_h3.drop(columns=["geometry", "id"]).groupby("ID", as_index=False).sum()

# 🚀 5. Unir los datos de carreteras al `df_agrupado`
df_agrupado = df_agrupado.merge(gdf_roads_summary, on="ID", how="left").fillna(0)

print(df_agrupado.head())

df_agrupado = df_agrupado.drop(columns=["num_POIs_y"], errors="ignore")  # Elimina la columna incorrecta
df_agrupado = df_agrupado.rename(columns={"num_POIs_x": "num_POIs"})  # Renombra para mantener consistencia

columnas_a_eliminar = ['CODE_2', 'POB1', 'OID_1','index_right']  # Lista de nombres de columnas a eliminar
df_agrupado = df_agrupado.drop(columns=columnas_a_eliminar)


# **1️ Seleccionar variables relevantes con sus pesos ajustados**
variables_mep = {
    # Población (25%)
    "POBTOT": 0.1,
    "POBFEM": 0.005128, "POBMAS": 0.005128, "P_12YMAS": 0.005128,
    "P_12YMAS_F": 0.005128, "P_12YMAS_M": 0.005128, "P_15YMAS": 0.005128, "P_18YMAS": 0.005128,
    "P_18YMAS_F": 0.005128, "P_18YMAS_M": 0.005128, "P_12A14": 0.005128, "P_12A14_F": 0.005128,
    "P_12A14_M": 0.005128, "P_15A17": 0.005128, "P_15A17_F": 0.005128, "P_15A17_M": 0.005128,
    "P_18A24": 0.005128, "P_18A24_F": 0.005128, "P_18A24_M": 0.005128, "P_60YMAS": 0.005128,
    "P_60YMAS_F": 0.005128, "P_60YMAS_M": 0.005128, "POB65_MAS": 0.005128, "PROM_HNV": 0.005128,
    "PSINDER": 0.005128, "PDER_SS": 0.005128, "PAFIL_IPRI": 0.005128, "TOTHOG": 0.005128,
    "POBHOG": 0.005128, "VIVTOT": 0.005128, "TVIVPAR": 0.005128, "VIVPAR_HAB": 0.005128,
    "VIVPAR_UT": 0.005128, "VPH_RADIO": 0.005128, "VPH_TV": 0.005128, "VPH_CEL": 0.005128,
    "VPH_INTER": 0.005128, "VPH_CVJ": 0.005128,

    # Tipología de Gasto (25%)
    "P_T_TG": 0.05, "PP_EURO_TG": 0.05,
    "CSP01_EURO": 0.0075, "CSP02_EURO": 0.0075, "CSP03_EURO": 0.0075, "CSP04_EURO": 0.0075,
    "CSP05_EURO": 0.0075, "CSP06_EURO": 0.0075, "CSP07_EURO": 0.0075, "CSP08_EURO": 0.0075,
    "CSP09_EURO": 0.0075, "CSP10_EURO": 0.0075, "CSP11_EURO": 0.0075, "CSP12_EURO": 0.0075,
    "CSP13_EURO": 0.0075, "CSP14_EURO": 0.0075, "CSP15_EURO": 0.0075, "CSP16_EURO": 0.0075,
    "CSP17_EURO": 0.0075, "CSP18_EURO": 0.0075, "CSP19_EURO": 0.0075, "CSP20_EURO": 0.0075,

    # Movilidad y Tráfico (25%)
    "num_motorway_OMR": 0.10,
    "num_trunk_OMR": 0.05,
    "num_primary_OMR": 0.05,
    "Metro": 0.015,
    "Metrobus": 0.015,
    "RTP": 0.015,
    "Ecobici": 0.015,
    "num_pedestrian_OMR": 0.05,

    # Puntos de Interés (25%)
    "num_POIs": 0.25
}

# **2️ Filtrar el DataFrame con solo las variables seleccionadas**
df_mep = df_agrupado[list(variables_mep.keys())].copy()

# **3️ Aplicar transformación logarítmica a variables con alta asimetría**
umbral_skew = 1.5
skew_values = df_mep.skew()
log_transform_vars = skew_values[skew_values > umbral_skew].index.tolist()

# Asegurar que las columnas sean numéricas y estén limpias
for col in log_transform_vars:
    df_mep[col] = pd.to_numeric(df_mep[col], errors='coerce').fillna(0)
    df_mep[col] = np.log1p(df_mep[col])  # log(1 + x)


df_mep[log_transform_vars] = np.log1p(df_mep[log_transform_vars])  # log(1 + x)

# **4️ Aplicar normalización con Z-score (StandardScaler)**
scaler = StandardScaler()
df_mep_normalizado = pd.DataFrame(scaler.fit_transform(df_mep), columns=df_mep.columns)

# **5️ Calcular la Métrica de Exposición Publicitaria (MEP) con los nuevos pesos**
df_agrupado["MEP"] = sum(df_mep_normalizado[col] * peso for col, peso in variables_mep.items())

# **6️ Normalizar nuevamente para que el MEP esté entre 0 y 1**
df_agrupado["MEP"] = (df_agrupado["MEP"] - df_agrupado["MEP"].min()) / (df_agrupado["MEP"].max() - df_agrupado["MEP"].min())
# **7️ Mostrar las primeras filas con la nueva métrica MEP calculada**
df_agrupado[["ID", "MEP"]].head()

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 10)
vmax = np.percentile(meps_normalizados, 90)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m


# 1️ Seleccionar el ID H3 con mayor valor en la métrica temática deseada
# (puedes cambiar 'MEP_comida' por otra métrica como 'MEP_universitarios', etc.)
hex_max = df_agrupado.sort_values("MEP", ascending=False).iloc[0]["ID"]

# 2️ Obtener el centro + sus 6 vecinos usando h3.k_ring()
hex_7_ids = list(h3.k_ring(hex_max, 1))  # devuelve 7 IDs H3

# 3️ Filtrar del GeoDataFrame los hexágonos correspondientes
df_hex7_1 = df_agrupado[df_agrupado["ID"].isin(hex_7_ids)].copy()

# 4️ Crear el mapa centrado en el centroide del hexágono principal
centroide = df_hex7_1[df_hex7_1["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_1 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 5️ Pintar los 7 hexágonos en rojo
for _, row in df_hex7_1.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "red",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_1)

# 6️ Mostrar el mapa
mapa_top7_1

def calcular_score_pois(pois_list):
    score = 0

    # Verifica que sea una lista válida
    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):  # Solo procesa si es un diccionario
                cat = str(poi.get('category', '')).lower()
                subcat = str(poi.get('subcategor', '')).lower()
                sublevel = str(poi.get('sublevel', '')).lower()

                if 'college and university' in cat or 'college and university' in subcat or 'college and university' in sublevel:
                    score += 7
                elif 'education' in cat or 'education' in subcat or 'education' in sublevel:
                    score += 4

    return score

df_agrupado["score_POIS_universitarios"] = df_agrupado["pois_data"].apply(calcular_score_pois)
print(df_agrupado["score_POIS_universitarios"].describe())


# 1️ Variables relevantes para el target "Estudiantes Universitarios"
variables_universitarios = {
    # Población
    "P_18A24": 0.05, "P_18A24_F": 0.02, "P_18A24_M": 0.02,
    "P_15A17": 0.02, "P_15A17_F": 0.01, "P_15A17_M": 0.01,
    
    "P_12YMAS": 0.02, "P_12YMAS_F": 0.01, "P_12YMAS_M": 0.01,
    "PNACOE": 0.02, "PNACOE_F": 0.01, "PNACOE_M": 0.01,
    "PROM_HNV": 0.02,
    
    # Movilidad
    "Metro": 0.03, "Metrobus": 0.03, "RTP": 0.02, "Ecobici": 0.02, "Trolebus": 0.02,

    # Gasto
    "CSP04_EURO": 0.02, "CSP05_EURO": 0.02, "CSP13_EURO": 0.03,
    "CSP14_EURO": 0.02, "CSP16_EURO": 0.02, "CSP17_EURO": 0.01, "CSP18_EURO": 0.02,
    

    # POIs
    "num_POIs": 0.01,
    "score_POIS_universitarios": 0.29,


    
}

# 2️ Subset del DataFrame solo con esas variables
df_uni = df_agrupado[list(variables_universitarios.keys())].copy()

# **3️ Aplicar transformación logarítmica a variables con alta asimetría**
umbral_skew = 1.5
skew_values = df_uni.skew()
log_transform_vars = skew_values[skew_values > umbral_skew].index.tolist()

# Asegurar que las columnas sean numéricas y estén limpias
for col in log_transform_vars:
    df_uni[col] = pd.to_numeric(df_uni[col], errors='coerce').fillna(0)
    df_uni[col] = np.log1p(df_uni[col])  # log(1 + x)


df_uni[log_transform_vars] = np.log1p(df_uni[log_transform_vars])  # log(1 + x)

# 4️ Estandarización con Z-score
scaler = StandardScaler()
df_uni_normalizado = pd.DataFrame(scaler.fit_transform(df_uni), columns=df_uni.columns)

# 5️ Calcular el MEP temático
df_agrupado["MEP_universitarios"] = sum(
    df_uni_normalizado[col] * peso for col, peso in variables_universitarios.items()
)

# 6️ Normalizar entre 0 y 1
df_agrupado["MEP_universitarios"] = (
    df_agrupado["MEP_universitarios"] - df_agrupado["MEP_universitarios"].min()
) / (
    df_agrupado["MEP_universitarios"].max() - df_agrupado["MEP_universitarios"].min() + 1e-9
)


# **7️ Mostrar las primeras filas con la nueva métrica MEP calculada**
df_agrupado[["ID", "MEP_universitarios"]].head()

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_universitarios"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
#meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_universitarios_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 10)
vmax = np.percentile(meps_normalizados, 90)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_universitarios_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Seleccionar el ID H3 con mayor valor en la métrica temática deseada
# (puedes cambiar 'MEP_comida' por otra métrica como 'MEP_universitarios', etc.)
hex_max = df_agrupado.sort_values("MEP_universitarios", ascending=False).iloc[0]["ID"]

# 2️ Obtener el centro + sus 6 vecinos usando h3.k_ring()
hex_7_ids = list(h3.k_ring(hex_max, 1))  # devuelve 7 IDs H3

# 3️ Filtrar del GeoDataFrame los hexágonos correspondientes
df_hex7_2 = df_agrupado[df_agrupado["ID"].isin(hex_7_ids)].copy()

# 4️ Crear el mapa centrado en el centroide del hexágono principal
centroide = df_hex7_2[df_hex7_2["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_2 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 5️ Pintar los 7 hexágonos en rojo
for _, row in df_hex7_2.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "yellow",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_2)

# 6️ Mostrar el mapa
mapa_top7_2

def calcular_score_pois_mascotas(pois_list):
    score = 0

    # Verifica que sea una lista válida
    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):  # Solo procesa si es un diccionario
                cat = str(poi.get('category', '')).lower()
                subcat = str(poi.get('subcategor', '')).lower()
                sublevel = str(poi.get('sublevel', '')).lower()

                if ('pet service' in cat or 'pet grooming service' in cat or 'pet supplies store' in cat or
                    'pet café' in cat or 'pet sitting and boarding service' in cat or 'veterinarian' in cat or
                    'pet service' in subcat or 'pet grooming service' in subcat or 'pet supplies store' in subcat or
                    'pet café' in subcat or 'pet sitting and boarding service' in subcat or 'veterinarian' in subcat or
                    'pet service' in sublevel or 'pet grooming service' in sublevel or 'pet supplies store' in sublevel or
                    'pet café' in sublevel or 'pet sitting and boarding service' in sublevel or 'veterinarian' in sublevel):
                    score += 10
                else:
                    score += 2

    return score

df_agrupado["score_POIS_mascotas"] = df_agrupado["pois_data"].apply(calcular_score_pois_mascotas)
print(df_agrupado["score_POIS_mascotas"].describe())

from sklearn.preprocessing import StandardScaler
import numpy as np

# 1️ Definir variables y pesos para el MEP temático
variables_mep_mascotas = {
    # Población (20%)
    "POBTOT": 0.05,
    "PROM_HNV": 0.05, "TOTHOG": 0.05, "POBHOG": 0.025, "VIVTOT": 0.025,

    # Gasto (25%)
    "CSP01_EURO": 0.05, "CSP05_EURO": 0.05, "CSP06_EURO": 0.0375,
    "CSP10_EURO": 0.0375, "CSP11_EURO": 0.0375, "CSP19_EURO": 0.0375,

    # Movilidad (45%)
    "num_pedestrian_OMR": 0.15, "num_path_OMR": 0.15, "num_footway_OMR": 0.05,
    "num_cycleway_OMR": 0.05, "num_residential_OMR": 0.05,

    # POIs (10%)
    "score_POIS_mascotas": 0.10
}

# 2️ Filtrar y copiar las columnas necesarias
df_mep_mascotas = df_agrupado[list(variables_mep_mascotas.keys())].copy()

# **3️ Aplicar transformación logarítmica a variables con alta asimetría**
umbral_skew = 1.5
skew_values = df_mep_mascotas.skew()
log_transform_vars = skew_values[skew_values > umbral_skew].index.tolist()

# Asegurar que las columnas sean numéricas y estén limpias
for col in log_transform_vars:
    df_mep_mascotas[col] = pd.to_numeric(df_mep_mascotas[col], errors='coerce').fillna(0)
    df_mep_mascotas[col] = np.log1p(df_mep_mascotas[col])  # log(1 + x)


df_mep_mascotas[log_transform_vars] = np.log1p(df_mep_mascotas[log_transform_vars])  # log(1 + x)

# 4️ Normalización Z-score
scaler = StandardScaler()
df_mep_mascotas_norm = pd.DataFrame(scaler.fit_transform(df_mep_mascotas), columns=df_mep_mascotas.columns)

# 5️ Calcular MEP específico
df_agrupado["MEP_mascotas"] = sum(
    df_mep_mascotas_norm[col] * peso for col, peso in variables_mep_mascotas.items()
)

# 6️ Normalizar entre 0 y 1
df_agrupado["MEP_mascotas"] = (df_agrupado["MEP_mascotas"] - df_agrupado["MEP_mascotas"].min()) / (
    df_agrupado["MEP_mascotas"].max() - df_agrupado["MEP_mascotas"].min()
)

# 7️ Verificar
df_agrupado[["ID", "MEP_mascotas"]].head()

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_mascotas"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_mascotas_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 3)
vmax = np.percentile(meps_normalizados, 97)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_mascotas_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Seleccionar el ID H3 con mayor valor en la métrica temática deseada
# (puedes cambiar 'MEP_comida' por otra métrica como 'MEP_universitarios', etc.)
hex_max = df_agrupado.sort_values("MEP_mascotas", ascending=False).iloc[0]["ID"]

# 2️ Obtener el centro + sus 6 vecinos usando h3.k_ring()
hex_7_ids = list(h3.k_ring(hex_max, 1))  # devuelve 7 IDs H3

# 3️ Filtrar del GeoDataFrame los hexágonos correspondientes
df_hex7_3 = df_agrupado[df_agrupado["ID"].isin(hex_7_ids)].copy()

# 4️ Crear el mapa centrado en el centroide del hexágono principal
centroide = df_hex7_3[df_hex7_3["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_3 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 5️ Pintar los 7 hexágonos en rojo
for _, row in df_hex7_3.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "green",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_3)

# 6️ Mostrar el mapa
mapa_top7_3

def calcular_score_pois_comida(pois_list):
    score = 0

    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):
                cat = str(poi.get('category', '')).lower()
                subcat = str(poi.get('subcategor', '')).lower()
                sublevel = str(poi.get('sublevel', '')).lower()

                # Comprobamos si contiene palabras clave de comida
                if 'restaurant' in cat or 'restaurant' in subcat or 'restaurant' in sublevel:
                    score += 10
                elif ('dining and drinking' in cat or 'dining and drinking' in subcat or 'dining and drinking' in sublevel or
                      'food and beverage service' in cat or 'food and beverage service' in subcat or 'food and beverage service' in sublevel or
                      'food and beverage retail' in cat or 'food and beverage retail' in subcat or 'food and beverage retail' in sublevel):
                    score += 7
                else:
                    score += 2  # Otros POIs no relacionados con comida

    return score

# Aplicar al DataFrame
df_agrupado["score_POIS_comida"] = df_agrupado["pois_data"].apply(calcular_score_pois_comida)

# Verificar estadísticas básicas
print(df_agrupado["score_POIS_comida"].describe())

# 1️ NUEVOS PESOS AJUSTADOS
variables_mep_comida = {
    # 🧍 Población (20%)
    "POBTOT": 0.05,
    "P_18YMAS": 0.05,
    "P_25YMAS": 0.05 if "P_25YMAS" in df_agrupado.columns else 0,

    # 💸 Gasto (35%)
    "CSP01_EURO": 0.275,
    "CSP02_EURO": 0.025,
    "CSP16_EURO": 0.025,
    "CSP18_EURO": 0.05,

    # 🚍 Movilidad (10%)
    "Metro": 0.02,
    "Metrobus": 0.02,
    "Ecobici": 0.01,
    "num_pedestrian_OMR": 0.03,
    "num_secondary_OMR": 0.02,

    # 📍 POIs (25%)
    "score_POIS_comida": 0.25
}

# 2️ FILTRAR Y VERIFICAR VARIABLES DISPONIBLES
variables_utilizadas = [var for var in variables_mep_comida.keys() if var in df_agrupado.columns]
df_mep = df_agrupado[variables_utilizadas].copy()

# 3️ TRANSFORMACIÓN LOG(1+x)
log_transform_vars = [v for v in variables_utilizadas if any(k in v for k in ["POB", "Metro", "bus", "num_", "score"])]
df_mep[log_transform_vars] = np.log1p(df_mep[log_transform_vars])

# 4️ NORMALIZACIÓN (Z-score)
scaler = StandardScaler()
df_mep_normalizado = pd.DataFrame(scaler.fit_transform(df_mep), columns=df_mep.columns)

# 5️ CÁLCULO DEL MEP TEMÁTICO
df_agrupado["MEP_comida"] = sum(
    df_mep_normalizado[col] * peso for col, peso in variables_mep_comida.items() if col in df_mep_normalizado.columns
)

# 6️ NORMALIZACIÓN FINAL ENTRE 0 Y 1
df_agrupado["MEP_comida"] = (
    df_agrupado["MEP_comida"] - df_agrupado["MEP_comida"].min()
) / (df_agrupado["MEP_comida"].max() - df_agrupado["MEP_comida"].min())

# 7️ REVISIÓN
df_agrupado[["ID", "MEP_comida"]].sort_values("MEP_comida", ascending=False).head()

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_comida"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_comida_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 1)
vmax = np.percentile(meps_normalizados, 99)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_comida_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Seleccionar el ID H3 con mayor valor en la métrica temática deseada
# (puedes cambiar 'MEP_comida' por otra métrica como 'MEP_universitarios', etc.)
hex_max = df_agrupado.sort_values("MEP_comida", ascending=False).iloc[0]["ID"]

# 2️ Obtener el centro + sus 6 vecinos usando h3.k_ring()
hex_7_ids = list(h3.k_ring(hex_max, 1))  # devuelve 7 IDs H3

# 3️ Filtrar del GeoDataFrame los hexágonos correspondientes
df_hex7_4 = df_agrupado[df_agrupado["ID"].isin(hex_7_ids)].copy()

# 4️ Crear el mapa centrado en el centroide del hexágono principal
centroide = df_hex7_4[df_hex7_4["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_4 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 5️ Pintar los 7 hexágonos en rojo
for _, row in df_hex7_4.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "orange",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_4)

# 6️ Mostrar el mapa
mapa_top7_4

def calcular_score_pois_abuelos(pois_list):
    score = 0

    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):
                cat = str(poi.get('category', '')).lower()
                subcat = str(poi.get('subcategor', '')).lower()
                sublevel = str(poi.get('sublevel', '')).lower()

                # Score 10
                if 'residential building' in cat or 'drugstore' in cat or 'pharmacy' in cat \
                   or 'residential building' in subcat or 'drugstore' in subcat or 'pharmacy' in subcat \
                   or 'residential building' in sublevel or 'drugstore' in sublevel or 'pharmacy' in sublevel:
                    score += 10

                # Score 6
                elif 'natural park' in cat or 'national park' in cat \
                     or 'natural park' in subcat or 'national park' in subcat \
                     or 'natural park' in sublevel or 'national park' in sublevel:
                    score += 6

                # Resto
                else:
                    score += 2

    return score

# Aplicar la función
df_agrupado["score_POIS_abuelos"] = df_agrupado["pois_data"].apply(calcular_score_pois_abuelos)

# Revisar estadísticas
print(df_agrupado["score_POIS_abuelos"].describe())

# 1️ Variables y pesos
variables_mep_abuelos = {
    # 👵 Población (35%)
    "P_60YMAS": 0.157,
    "POB65_MAS": 0.05, "PROM_HNV": 0.04,
    

    # 💊 Gasto (10%)
    "CSP01_EURO": 0.025, "CSP12_EURO": 0.025,
    

    # 🚶 Movilidad (15%)
    "num_pedestrian_OMR": 0.075,
    "num_residential_OMR": 0.075,

    # 📍 POIs (30%)
    "score_POIS_abuelos": 0.30
}

# 2️ Filtrar columnas
df_mep_abuelos = df_agrupado[list(variables_mep_abuelos.keys())].copy()

# 3️ Transformaciones logarítmicas si es necesario
log_vars_abuelos = ["score_POIS_abuelos"]
df_mep_abuelos[log_vars_abuelos] = np.log1p(df_mep_abuelos[log_vars_abuelos])

# 4️ Normalización Z-score
scaler_abuelos = StandardScaler()
df_mep_abuelos_norm = pd.DataFrame(scaler_abuelos.fit_transform(df_mep_abuelos), columns=df_mep_abuelos.columns)

# 5️ Calcular el MEP
df_agrupado["MEP_abuelos"] = sum(df_mep_abuelos_norm[col] * peso for col, peso in variables_mep_abuelos.items())

# 6️ Normalizar entre 0 y 1
df_agrupado["MEP_abuelos"] = (df_agrupado["MEP_abuelos"] - df_agrupado["MEP_abuelos"].min()) / (df_agrupado["MEP_abuelos"].max() - df_agrupado["MEP_abuelos"].min())

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_abuelos"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_abuelos_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 2)
vmax = np.percentile(meps_normalizados, 98)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_abuelos_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Iterar sobre todos los hexágonos y buscar el grupo de 7 con mayor MEP promedio
hex_ids = df_agrupado["ID"].astype(str).tolist()

mejor_isla = None
max_promedio = -1

for h in hex_ids:
    try:
        vecinos = list(h3.k_ring(h, 1))  # centro + 6 vecinos
        vecinos_df = df_agrupado[df_agrupado["ID"].astype(str).isin(vecinos)]
        
        if len(vecinos_df) == 7:  # Asegurar que todos estén presentes
            promedio = vecinos_df["MEP_abuelos"].mean()
            if promedio > max_promedio:
                max_promedio = promedio
                mejor_isla = vecinos_df
    except Exception:
        continue

# 2️ Guardar la mejor isla en df_hex7_5
df_hex7_5 = mejor_isla.copy()

# 3️ Centrar el mapa en el centroide del hexágono central (el más alto del grupo)
hex_max = df_hex7_5.sort_values("MEP_abuelos", ascending=False).iloc[0]["ID"]
centroide = df_hex7_5[df_hex7_5["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_5 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 4️ Pintar los 7 hexágonos
for _, row in df_hex7_5.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "gray",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_5)

# 5️ Mostrar el mapa
mapa_top7_5

def calcular_score_pois_fisioterapia(pois_list):
    score = 0
    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):
                cat = str(poi.get('category', '')).lower()
                subcat = str(poi.get('subcategor', '')).lower()
                sublevel = str(poi.get('sublevel', '')).lower()

                # 🎯 Muy relevantes (10 puntos)
                if any(x in (cat, subcat, sublevel) for x in [
                    'gym and studio', 'gymnastics', 'gymnastics center',
                    'college gym', 'physical therapy clinic'
                ]):
                    score += 10

                # ✅ Relevantes (5 puntos)
                elif 'health and medicine' in (cat, subcat, sublevel):
                    score += 5

                # ➖ Poco relevantes (2 puntos)
                else:
                    score += 2
    return score

# Aplicar al DataFrame
df_agrupado["score_POIS_fisioterapia"] = df_agrupado["pois_data"].apply(calcular_score_pois_fisioterapia)

# Comprobación
print(df_agrupado["score_POIS_fisioterapia"].describe())

 #1️ Definir las variables y pesos para el target Fisioterapia
variables_mep_fisio = {
    # 🧍‍♂️ Población (30%)
    "P_15A49_F": 0.05,     # Mujeres adultas activas
    "P_18YMAS": 0.05,      # Población adulta general
    "P_60YMAS": 0.05,      # Personas mayores con más necesidad
    "POBTOT": 0.05,        # Población total (base)
    "PROM_HNV": 0.05,      # Hijos nacidos vivos (podría indicar actividad física previa o necesidad de cuidado)


    # 💳 Tipología de gasto (20%)
    "CSP12_EURO": 0.17,    # Productos médicos
  
    "CSP19_EURO": 0.04,    # Cuidado personal
    "CSP16_EURO": 0.05,    # Servicios de recreación/cultura (actividades de bienestar)

    # 🚍 Transporte público (10%)
    "Metro": 0.015, "Metrobus": 0.015, "RTP": 0.015, "Ecobici": 0.015,
    "Trolebus": 0.01, "Cablebus": 0.01,


    # 🏋️‍♀️ POIs orientados a fisioterapia (30%)
    "score_POIS_fisioterapia": 0.39
}

# 2️ Filtrar DataFrame solo con las variables seleccionadas
df_fisio = df_agrupado[list(variables_mep_fisio.keys())].copy()

# 3️ Aplicar log(1 + x) a las variables que pueden tener valores extremos
log_transform_vars_fisio = [
    "POBTOT", "P_18YMAS", "P_60YMAS", "score_POIS_fisioterapia",
    "Metro", "Metrobus", "RTP", "Ecobici", "Trolebus", "Cablebus"
]
df_fisio[log_transform_vars_fisio] = np.log1p(df_fisio[log_transform_vars_fisio])

# 4️ Normalización con Z-score
scaler_fisio = StandardScaler()
df_fisio_normalizado = pd.DataFrame(
    scaler_fisio.fit_transform(df_fisio),
    columns=df_fisio.columns
)

# 5️ Calcular MEP_fisioterapia con los pesos definidos
df_agrupado["MEP_fisioterapia"] = sum(
    df_fisio_normalizado[col] * peso
    for col, peso in variables_mep_fisio.items()
)

# 6️ Normalizar MEP final entre 0 y 1
df_agrupado["MEP_fisioterapia"] = (
    df_agrupado["MEP_fisioterapia"] - df_agrupado["MEP_fisioterapia"].min()
) / (
    df_agrupado["MEP_fisioterapia"].max() - df_agrupado["MEP_fisioterapia"].min()
)

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_fisioterapia"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_fisioterapia_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 1)
vmax = np.percentile(meps_normalizados, 99)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_fisioterapia_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Iterar sobre todos los hexágonos y buscar el grupo de 7 con mayor MEP promedio
hex_ids = df_agrupado["ID"].astype(str).tolist()

mejor_isla = None
max_promedio = -1

for h in hex_ids:
    try:
        vecinos = list(h3.k_ring(h, 1))  # centro + 6 vecinos
        vecinos_df = df_agrupado[df_agrupado["ID"].astype(str).isin(vecinos)]
        
        if len(vecinos_df) == 7:  # Asegurar que todos estén presentes
            promedio = vecinos_df["MEP_fisioterapia"].mean()
            if promedio > max_promedio:
                max_promedio = promedio
                mejor_isla = vecinos_df
    except Exception:
        continue

# 2️ Guardar la mejor isla en df_hex7_5
df_hex7_6 = mejor_isla.copy()

# 3️ Centrar el mapa en el centroide del hexágono central (el más alto del grupo)
hex_max = df_hex7_6.sort_values("MEP_fisioterapia", ascending=False).iloc[0]["ID"]
centroide = df_hex7_6[df_hex7_6["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_6 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 4️ Pintar los 7 hexágonos
for _, row in df_hex7_6.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "yellow",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_6)

# 5️ Mostrar el mapa
mapa_top7_6

def calcular_score_pois_mecanicos(pois_list):
    score = 0
    categorias_mecanicos = {
        "car dealership",
        "rental car location",
        "automotive service"
    }

    if isinstance(pois_list, list):
        for poi in pois_list:
            if isinstance(poi, dict):
                cat = str(poi.get("category", "")).lower()
                subcat = str(poi.get("subcategor", "")).lower()
                sublevel = str(poi.get("sublevel", "")).lower()

                if (
                    cat in categorias_mecanicos
                    or subcat in categorias_mecanicos
                    or sublevel in categorias_mecanicos
                ):
                    score += 10
                else:
                    score += 0  # explícito, aunque no es necesario

    return score

# Aplicar el cálculo al DataFrame
df_agrupado["score_POIS_mecanicos"] = df_agrupado["pois_data"].apply(calcular_score_pois_mecanicos)

# Ver estadísticos
print(df_agrupado["score_POIS_mecanicos"].describe())

# 1️ Definir pesos para el MEP de mecánicos
variables_mep_mecanicos = {
    # 🛣️ Carreteras (50%)
    "num_motorway_OMR": 0.15,  # Autopistas
    "num_trunk_OMR": 0.05,     # Vías troncales
    "num_primary_OMR": 0.10,   # Carreteras primarias
   # Carreteras de servicio

    # 🧭 POIs (30%) - Score calculado
    "score_POIS_mecanicos": 0.20,

    # 👥 Población (20%)
    
    "VPH_AUTOM": 0.3,  # Adultos
    "VPH_MOTO": 0.3,    # Total de hogares (tienen vehículo)
   # Total viviendas (posible relación con posesión de coche)
}

# 2️ Seleccionar solo las variables relevantes
df_mep_mecanicos = df_agrupado[list(variables_mep_mecanicos.keys())].copy()

# 3️ Aplicar log(1 + x) a variables con valores extremos
log_vars = [
    "num_motorway_OMR", "num_trunk_OMR", "num_primary_OMR", "score_POIS_mecanicos"
]
df_mep_mecanicos[log_vars] = np.log1p(df_mep_mecanicos[log_vars])

# 4️ Normalizar con StandardScaler
scaler = StandardScaler()
df_mep_mecanicos_normalizado = pd.DataFrame(
    scaler.fit_transform(df_mep_mecanicos),
    columns=df_mep_mecanicos.columns
)

# 5️ Calcular el MEP temático
df_agrupado["MEP_mecanicos"] = sum(
    df_mep_mecanicos_normalizado[col] * peso
    for col, peso in variables_mep_mecanicos.items()
)

# 6️ Escalar el MEP entre 0 y 1
df_agrupado["MEP_mecanicos"] = (
    (df_agrupado["MEP_mecanicos"] - df_agrupado["MEP_mecanicos"].min()) /
    (df_agrupado["MEP_mecanicos"].max() - df_agrupado["MEP_mecanicos"].min())
)

# 7️ Vista previa
df_agrupado[["ID", "MEP_mecanicos"]].sort_values("MEP_mecanicos", ascending=False).head(10)
print(df_agrupado["MEP_mecanicos"].describe())

# Asegurar que df_agrupado tiene geometría válida
df_agrupado = df_agrupado[df_agrupado.geometry.notnull()]

# Normalizar el MEP para mejorar la visualización en el mapa
meps = df_agrupado["MEP_mecanicos"].values

# Aplicar transformación logarítmica para mejorar la diferenciación de valores bajos
meps = np.log1p(meps)  # log(1 + MEP) para evitar valores negativos

# Normalizar entre 0 y 1
meps_normalizados = (meps - np.min(meps)) / (np.max(meps) - np.min(meps) + 1e-9)
df_agrupado["MEP_mecanicos_norm"] = meps_normalizados


# Crear mapa centrado en la Ciudad de México
m = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

vmin = np.percentile(meps_normalizados, 10)
vmax = np.percentile(meps_normalizados, 90)


# Definir una escala de colores más contrastante
colormap = cm.LinearColormap(
    colors=["darkblue", "blue", "green", "yellow", "orange", "black"], 
    vmin=vmin, vmax=vmax
)

# Agregar hexágonos H3 con colores según el MEP
for _, row in df_agrupado.iterrows():
    color = colormap(row["MEP_mecanicos_norm"])  # Asignar color basado en MEP normalizado
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, color=color: {
            "fillColor": color, 
            "color": "black", 
            "weight": 0.3,  # Reducimos el peso del contorno para destacar color
            "fillOpacity": 0.7  # Aumentamos la opacidad para hacer el color más prominente
        }
    ).add_to(m)

# Agregar la leyenda de colores
colormap.caption = "Métrica de Exposición Publicitaria (MEP)"
m.add_child(colormap)

# Mostrar el mapa
m

# 1️ Iterar sobre todos los hexágonos y buscar el grupo de 7 con mayor MEP promedio
hex_ids = df_agrupado["ID"].astype(str).tolist()

mejor_isla = None
max_promedio = -1

for h in hex_ids:
    try:
        vecinos = list(h3.k_ring(h, 1))  # centro + 6 vecinos
        vecinos_df = df_agrupado[df_agrupado["ID"].astype(str).isin(vecinos)]
        
        if len(vecinos_df) == 7:  # Asegurar que todos estén presentes
            promedio = vecinos_df["MEP_mecanicos"].mean()
            if promedio > max_promedio:
                max_promedio = promedio
                mejor_isla = vecinos_df
    except Exception:
        continue

# 2️ Guardar la mejor isla en df_hex7_5
df_hex7_7 = mejor_isla.copy()

# 3️ Centrar el mapa en el centroide del hexágono central (el más alto del grupo)
hex_max = df_hex7_7.sort_values("MEP_mecanicos", ascending=False).iloc[0]["ID"]
centroide = df_hex7_7[df_hex7_7["ID"] == hex_max].geometry.iloc[0].centroid
mapa_top7_7 = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, tiles='cartodbpositron')

# 4️ Pintar los 7 hexágonos
for _, row in df_hex7_7.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature: {
            "fillColor": "cyan",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    ).add_to(mapa_top7_7)

# 5️ Mostrar el mapa
mapa_top7_7


# Crear mapa base centrado en CDMX
m_final = folium.Map(location=[19.4326, -99.1332], zoom_start=12, tiles="cartodbpositron")

# Diccionario con tus GeoDataFrames (islas)
mapas = {
    "General": df_hex7_1,
    "Universitarios": df_hex7_2,
    "Personas con mascotas": df_hex7_3,
    "Amantes de la comida": df_hex7_4,
    "Abuelos": df_hex7_5,
    "Fisioterapia" : df_hex7_6,
    "Mecánico": df_hex7_7,
}

# Colores personalizados por categoría
colores = {
    "General": "red",
    "Universitarios": "blue",
    "Personas con mascotas": "green",
    "Amantes de la comida": "orange",
    "Abuelos" : "gray",
    "Fisioterapia": "yellow",
    "Mecánico": "cyan"
}

# Añadir cada grupo de hexágonos al mapa
for nombre, gdf in mapas.items():
    capa = folium.FeatureGroup(name=nombre, show=True)
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry,
            style_function=lambda feature, color=colores[nombre]: {
                "fillColor": color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(f"{nombre} - ID: {row['ID']}")
        ).add_to(capa)
    capa.add_to(m_final)

# 🔲 Añadir contorno exterior de toda la malla
contorno = unary_union(df_agrupado.geometry)
gdf_contorno = gpd.GeoDataFrame(geometry=[contorno], crs="EPSG:4326")

folium.GeoJson(
    gdf_contorno,
    style_function=lambda feature: {
        "fillColor": "none",
        "color": "black",
        "weight": 2,
        "fillOpacity": 0,
    },
    tooltip="Límite de la malla"
).add_to(m_final)

# Control de capas para alternar visualización
folium.LayerControl(collapsed=False).add_to(m_final)

# Mostrar mapa
m_final