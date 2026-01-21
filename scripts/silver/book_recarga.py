from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import expr, col, round, avg, max, min, sum, count, when, lit
from itertools import product
from pyspark.sql.types import StructType, StructField, StringType, LongType
from delta import *



agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")


# MAGIC %md
# MAGIC ## Leitura dos dados na camada silver


path = "hackathon2025.silver.base_recarga"

## rodar o arquivo 6 vezes de 202410 até 202503
from datetime import datetime
from dateutil.relativedelta import relativedelta
#202410
data_exec_inicial = 202410

# converte YYYYMM -> date
data_dt = datetime.strptime(str(data_exec_inicial), "%Y%m")

# subtrai 12 meses
data_exec_final = int((data_dt - relativedelta(months=12)).strftime("%Y%m"))
data_exec_final


from pyspark.sql.functions import col
df_book_recarga_01 = (
    spark.read
         .table(path)
         .filter((col("SAFRA") == data_exec_inicial) & (col("COD_PLATAFORMA_ATU") == "PREPG"))
         .select("NUM_CPF")
         .distinct()
)
df_book_recarga_01.createOrReplaceTempView("df_book_recarga_01")



df_book_recarga_02 = (
    spark.read
         .table(path)
         .filter(
             (col("SAFRA") >= data_exec_final) &
             (col("SAFRA") <= data_exec_inicial) &
             (col("COD_PLATAFORMA_ATU") == "PREPG")
         )
)



# MAGIC %md
# MAGIC ## Criando flag de janela


df_book_recarga_02.createOrReplaceTempView("df_transacoes")

df_temp_01 = spark.sql("""
WITH base AS (
    SELECT
        *,
        TO_DATE(CONCAT(SAFRA, '01'), 'yyyyMMdd') AS data_dt
    FROM df_transacoes
)

SELECT
    *,


    -- JANELAS TEMPORAIS

    CASE
        WHEN data_dt BETWEEN
             ADD_MONTHS(MAX(data_dt) OVER (PARTITION BY NUM_CPF), -1)
             AND MAX(data_dt) OVER (PARTITION BY NUM_CPF)
        THEN 1 ELSE 0
    END AS U1M,

    CASE
        WHEN data_dt BETWEEN
             ADD_MONTHS(MAX(data_dt) OVER (PARTITION BY NUM_CPF), -3)
             AND MAX(data_dt) OVER (PARTITION BY NUM_CPF)
        THEN 1 ELSE 0
    END AS U3M,

    CASE
        WHEN data_dt BETWEEN
             ADD_MONTHS(MAX(data_dt) OVER (PARTITION BY NUM_CPF), -6)
             AND MAX(data_dt) OVER (PARTITION BY NUM_CPF)
        THEN 1 ELSE 0
    END AS U6M,

    CASE
        WHEN data_dt BETWEEN
             ADD_MONTHS(MAX(data_dt) OVER (PARTITION BY NUM_CPF), -9)
             AND MAX(data_dt) OVER (PARTITION BY NUM_CPF)
        THEN 1 ELSE 0
    END AS U9M,

    CASE
        WHEN data_dt BETWEEN
             ADD_MONTHS(MAX(data_dt) OVER (PARTITION BY NUM_CPF), -12)
             AND MAX(data_dt) OVER (PARTITION BY NUM_CPF)
        THEN 1 ELSE 0
    END AS U12M

FROM base
ORDER BY NUM_CPF, SAFRA

""")


df_temp_01.createOrReplaceTempView("df_temp_01")


# MAGIC %md
# MAGIC ## Criando variáveis explicativas de primeira camada



from pyspark.sql.functions import expr, col, round, avg, max, min, sum, count, when, lit
from itertools import product

# Definição das variáveis
coluna_chave = "NUM_CPF"
colunas_flags = ['U1M', 'U3M', 'U6M', 'U9M', 'U12M']

# Lista de colunas de valores
colunas_valores = [
    'VAL_CREDITO_INSERIDO', 'VAL_BONUS', 'VAL_REAL', 'VALOR_SOS'
]

# Configuração dos indicadores
indicadores_config = {
    'FLAG_SOS': {'alias': 'FLAG_SOS', 'valores': [1]},
}


def gerar_sql_dinamico():
    selects = ["NUM_CPF"]
    
    # Agregações básicas
    for flag in colunas_flags:
        for valor in colunas_valores:
            selects.append(f"round(avg(case when {flag} = 1 then {valor} else NULL end), 2) as VL_MED_{flag}_{valor}_RECARGA")
            selects.append(f"round(max(case when {flag} = 1 then {valor} else NULL end), 2) as VL_MAX_{flag}_{valor}_RECARGA")
            selects.append(f"round(min(case when {flag} = 1 then {valor} else NULL end), 2) as VL_MIN_{flag}_{valor}_RECARGA")
            selects.append(f"round(sum(case when {flag} = 1 then {valor} else NULL end), 2) as VL_TOT_{flag}_{valor}_RECARGA")
            selects.append(f"round(count(case when {flag} = 1 then {valor} else NULL end), 2) as VL_QT_{flag}_{valor}_RECARGA")
    
    # Agregações com indicadores
    for indicador, info in indicadores_config.items():
        for valor_indicador in info['valores']:
            for flag in colunas_flags:
                for valor in colunas_valores:
                    alias = info['alias']
                    selects.append(f"round(avg(case when {flag} = 1 and {indicador} = '{valor_indicador}' then {valor} else NULL end), 2) as VL_MED_{flag}_{alias}_{valor_indicador}_{valor}_RECARGA")
                    selects.append(f"round(max(case when {flag} = 1 and {indicador} = '{valor_indicador}' then {valor} else NULL end), 2) as VL_MAX_{flag}_{alias}_{valor_indicador}_{valor}_RECARGA")
                    selects.append(f"round(min(case when {flag} = 1 and {indicador} = '{valor_indicador}' then {valor} else NULL end), 2) as VL_MIN_{flag}_{alias}_{valor_indicador}_{valor}_RECARGA")                    
                    selects.append(f"round(count(case when {flag} = 1 and {indicador} = '{valor_indicador}' then {valor} else NULL end), 2) as VL_QT_{flag}_{alias}_{valor_indicador}_{valor}_RECARGA")
                    selects.append(f"round(sum(case when {flag} = 1 and {indicador} = '{valor_indicador}' then {valor} else NULL end), 2) as VL_TOT_{flag}_{alias}_{valor_indicador}_{valor}_RECARGA")                    
    
    sql_query = f"""
    SELECT
        {', '.join(selects)}
    FROM df_temp_01
    GROUP BY NUM_CPF
    ORDER BY NUM_CPF
    """
    
    return sql_query

# Executar SQL dinâmico
sql_dinamico = gerar_sql_dinamico()

df_temp_02 = spark.sql(sql_dinamico)

df_temp_02.createOrReplaceTempView("df_temp_02")



# MAGIC %md
# MAGIC ## Criando variáveis explicativas de segunda camada



def add_temporal_ratio_columns(
    df,
    base_prefix="VL_MED",
    ratio_prefix="RAZ_MED",
    windows=("U1M", "U3M", "U6M", "U9M", "U12M"),
    suffix="_RECARGA"
):
    """
    Função que mantém todas as colunas originais do DataFrame
    e adiciona novas colunas de razão temporal entre janelas consecutivas.

    Exemplo de razão criada:
    RAZ_MED_U1M_U3M_FAT_RECARGA = VL_MED_U1M_FAT_RECARGA / VL_MED_U3M_FAT_RECARGA
    """

    # Cria uma lista com todas as colunas originais do DataFrame
    # Isso garante que nenhuma coluna existente será removida
    base_cols = [F.col(c) for c in df.columns]

    # Lista que armazenará as expressões das novas colunas de razão
    ratio_exprs = []

    # Conjunto com os nomes das colunas do DataFrame
    # Usado para checar rapidamente se uma coluna existe
    df_cols = set(df.columns)

    # Gera pares de janelas consecutivas
    # Exemplo: (U1M, U3M), (U3M, U6M), ...
    window_pairs = list(zip(windows[:-1], windows[1:]))

    # Percorre cada par de janelas (numerador e denominador)
    for num_win, den_win in window_pairs:

        # Percorre todas as colunas do DataFrame
        for col_num in df.columns:

            # Verifica se a coluna pertence à janela do numerador
            # e segue o padrão: VL_MED_<JANELA>_<FEATURE>
            if not col_num.startswith(f"{base_prefix}_{num_win}_"):
                continue

            # Deriva o nome da coluna do denominador
            # Substitui a janela do numerador pela janela do denominador
            col_den = col_num.replace(
                f"{base_prefix}_{num_win}_",
                f"{base_prefix}_{den_win}_"
            )

            # Se a coluna do denominador não existir, ignora
            if col_den not in df_cols:
                continue

            # Extrai o nome da feature base
            # Remove prefixo (VL_MED_<JANELA>_) e o sufixo (_ATRASO)
            feature = (
                col_num
                .replace(f"{base_prefix}_{num_win}_", "")
                .replace(suffix, "")
            )

            # Define o nome da nova coluna de razão temporal
            # Exemplo: RAZ_MED_U1M_U3M_FAT_ATRASO
            ratio_name = (
                f"{ratio_prefix}_{num_win}_{den_win}_"
                f"{feature}{suffix}"
            )

            # Cria a expressão da razão temporal
            # Realiza a divisão apenas quando o denominador é diferente de zero
            # Caso contrário, retorna NULL
            ratio_exprs.append(
                F.when(
                    F.col(col_den) != 0,
                    F.col(col_num) / F.col(col_den)
                ).alias(ratio_name)
            )

    # Retorna o DataFrame mantendo todas as colunas originais
    # e adicionando as novas colunas de razão temporal
    return df.select(*base_cols, *ratio_exprs)


# Aplica a função ao DataFrame anterior, gerando um novo DataFrame enriquecido
df_temp_03 = add_temporal_ratio_columns(df_temp_02)


df_temp_03.createOrReplaceTempView("df_temp_03")


# MAGIC %md
# MAGIC ## Join entre os CPF's da safra ref e o book criado



df_temp_04 = df_book_recarga_01.alias("t1") \
    .join(df_temp_03.alias("t2"), "NUM_CPF", "left") \
    .withColumn("SAFRA", lit(data_exec_inicial)) \
    .withColumn("DATPROC", lit(dthproc))

df_temp_04.createOrReplaceTempView("df_temp_04")




silver_table = "hackathon2025.silver.book_recarga"



if spark.catalog.tableExists(silver_table):

    delta_table = DeltaTable.forName(spark, silver_table)
    (
        delta_table.alias("t")
        .merge(
            df_temp_04.alias("s"),
            "t.NUM_CPF = s.NUM_CPF AND t.SAFRA = s.SAFRA"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:

    df_temp_04.write \
        .format("delta") \
        .option("overwriteSchema", "true") \
        .mode("overwrite") \
        .saveAsTable(silver_table)




name = "book_recarga"

df_controle = spark.sql("""
    SELECT
        '{name_table}'        AS nome_tabela,
        SAFRA                 AS safra,
        COUNT(*)              AS qtd_registros,
        current_timestamp()   AS datproc
    FROM df_temp_04
    GROUP BY SAFRA
    ORDER BY SAFRA
""".format(name_table=name))



silver_table_controle = "hackathon2025.silver.controle"
if not spark.catalog.tableExists(silver_table_controle):
    print("Tabela silver_controle não existe. Criando...")

    (
        df_controle
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(silver_table_controle)  
    )

    print("Tabela silver_controle criada com sucesso")

else:
    print("Tabela de controle existe. Inserindo novo registro...")

    delta_silver = DeltaTable.forName(spark, silver_table_controle)

    (
        df_controle
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(silver_table_controle)
    )
    print("Dados inseridos com sucesso...")


