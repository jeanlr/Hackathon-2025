

# MAGIC %pip install category_encoders scikit-learn matplotlib pandas statsmodels shap feature_engine lightgbm optuna seaborn numpy 


# MAGIC %md
# MAGIC ## Importando bibliotecas



import numpy as np
import pandas as pd
import seaborn as sns
from gold.util import *
import matplotlib.pyplot as plt
from warnings import filterwarnings
from pyspark.sql.functions import col


df_bureau = spark.read.table("hackathon2025.gold.one_big_table_publico") \
    .filter((col("FLAG_INSTALACAO") == 1) & (col("Flag_mig2") == "PRE"))


df_publico = df_bureau.toPandas()
df_publico.head()


# MAGIC %md
# MAGIC ## Aplicando função de CEP para encontrar as regiões do país


def classifica_regiao_cep(cep):
    if pd.isna(cep):
        return None

    # garante string, remove caracteres e pega os 3 primeiros dígitos
    cep_str = ''.join(filter(str.isdigit, str(cep)))
    if len(cep_str) < 3:
        return None

    prefixo = int(cep_str[:3])

    if 0 <= prefixo <= 399:
        return "Sudeste"
    elif 400 <= prefixo <= 659:
        return "Nordeste"
    elif 660 <= prefixo <= 699:
        return "Norte"
    elif 700 <= prefixo <= 799:
        return "Centro-Oeste"
    elif 800 <= prefixo <= 999:
        return "Sul"
    else:
        return None


df_publico['regiao'] = df_publico['CEP_3_digitos'].apply(classifica_regiao_cep)


df_publico.head()


# MAGIC %md
# MAGIC ## Analisando volumetria de SAFRA e volumetria de público


df_publico_02 = df_publico.copy()
df_publico_02 = df_publico_02.drop(columns=['DATPROC'])


resultado_volumetria = df_publico_02.groupby('SAFRA').agg({'FPD': 'mean', 'SAFRA': 'count'}).rename(columns={'SAFRA': 'Volume'}).reset_index()
resultado_volumetria.columns = ['Safra (AAAA)', 'Taxa_de_Evento', 'Volume']

resultado_volumetria


df_tx_evento = plot_tx_event_volume_safra(df_publico_02,
                                              target='FPD',
                                              safra='SAFRA',
                                              ymax_volume=260000, ymax_taxa_evento=35)


# MAGIC %md
# MAGIC ## Analisando o público da Safra 2025-03


# DBTITLE 1,Cell 11
df_publico_03 = df_publico_02.copy()
df_publico_03 = df_publico_03.query('SAFRA == "202503"')


df_publico_03.head()


# MAGIC %md
# MAGIC ## Verificando os metadados


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


metadados = generate_metadata(df=df_publico_03, targets=['FPD', 'NUM_CPF', 'SAFRA', 'FLAG_INSTALACAO', 'PROD', 'flag_mig2'], orderby='PC_NULOS')
metadados


# MAGIC %md
# MAGIC ## Excluindo variáveis com mais de 75% de nulos


missing_cutoff = 70

drop_vars_nulos = metadados[(metadados['PC_NULOS'] >= missing_cutoff)]
lista_drop_vars = list(drop_vars_nulos.FEATURE.values)

print('Variáveis que serão excluídas por alto percentual de nulos', lista_drop_vars)

df_publico_03 = df_publico_03.drop(axis=1, columns=lista_drop_vars)
df_publico_03.shape


# MAGIC %md
# MAGIC ## Input de missing numérico


df_publico_03, means = fillna_numeric(df_publico_03)


# MAGIC %md
# MAGIC ## Input de missing categórico


df_publico_03, modes = fillna_categorical(df_publico_03)


# MAGIC %md
# MAGIC ## Entendimento do Target



target_counts = df_publico_03['FPD'].value_counts()

# Calculando percentuais de FPD
percent_n = (target_counts[0] / len(df_publico_03)) * 100
percent_s = (target_counts[1] / len(df_publico_03)) * 100

# Criando dataframe
total_por_target = pd.DataFrame({'target': ['Nao', 'Sim'], 'Percentual': [percent_n, percent_s]})

#Definição de cores
cores = {'Nao': 'lightgrey', 'Sim': 'salmon'}

# Plot do gráfico
ax = sns.barplot(x='target', y='Percentual', hue='target', data=total_por_target, palette=cores, dodge=False)

# Adicionando valores de cada barra
for p in ax.patches:
    ax.annotate("{:.2f}%".format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.xlabel('target')
plt.ylabel('Percentual')
plt.title('Percentaul FPD')
plt.show()


# MAGIC %md
# MAGIC ## Análises univariadas e bivariadas


# DBTITLE 1,Cell 20
import math
# Variáveis de interesse
target = 'FPD'
df_publico_04 = df_publico_03.copy()
df_publico_04 = df_publico_04.drop(columns=['NUM_CPF', 'SAFRA', 'FLAG_INSTALACAO', 'PROD', 'flag_mig2'])

boxplots_var_num(df_publico_04)


histograms_var_num(df_publico_04)


plot_categorical_frequency_pt(df_publico_04, corte_cardinalidade=10, graficos_por_linha=2)


kdeplots_var_num_target(df_publico_04, 'FPD')
plt.show()


plot_cat_vs_target_percentage(df_publico_04, 'FPD', cutoff=10)
plt.show()


# MAGIC %md
# MAGIC # Análise do Público
# MAGIC
# MAGIC ## Quantidade de inadimplentes VS adimplentes


pod_count_categorias(df_publico_04, ['FPD'])


# MAGIC %md
# MAGIC ## Idade


df_publico_04['IDADE'].describe().round(2)


# MAGIC %md
# MAGIC ## Score 01


df_publico_04['SCORE_01'].describe().round(2)


# MAGIC %md
# MAGIC ## Score 02


df_publico_04['SCORE_02'].describe().round(2)


# MAGIC %md
# MAGIC ## Região do país


pod_count_categorias(df_publico_04, ['regiao'])


resultado = (
    df_publico_04
        .groupby('regiao', dropna=False)
        .agg(
            volume=('regiao', 'count'),
            fpd_media=('FPD', 'mean')
        )
        .reset_index()
        .sort_values(by='fpd_media', ascending=False)
)

resultado


# MAGIC %md
# MAGIC Após a análise foi concluído que a região norte e nordeste possuem aproximadamente 2.p.p acima da média geral de FPD


# MAGIC %md
# MAGIC ## Status Receita Federal


pod_count_categorias(df_publico_04, ['STATUSRF'])


# MAGIC %md
# MAGIC ## var_24


pod_count_categorias(df_publico_04, ['var_24'])


resultado = (
    df_publico_04
        .groupby('var_09', dropna=False)
        .agg(
            volume=('var_09', 'count'),
            fpd_media=('FPD', 'mean')
        )
        .reset_index()
        .sort_values(by='var_09', ascending=False)
)

resultado