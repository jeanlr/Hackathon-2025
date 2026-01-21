from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz




agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")


df_base_dados_cadastrais = spark.read.parquet("/Volumes/hackathon2025/bronze/base_dados_cadastrais/")


df_base_dados_cadastrais.createOrReplaceTempView("raw_00")




lake = spark.sql(     
    """
        select
        
            cast(NUM_CPF as string) as NUM_CPF,
            try_cast(SAFRA as int) as SAFRA,
            try_cast(FLAG_INSTALACAO as int) as FLAG_INSTALACAO,
            try_cast(FPD as int) as FPD,
            cast(PROD as string) as PROD,
            cast(flag_mig2 as string) as flag_mig2,
            cast(STATUSRF as string) as STATUSRF,
            
            -- Data com tratamento para valores nulos/vazios usando to_date
            case 
                when trim(DATADENASCIMENTO) in ('null', 'NULL', '') then null
                else to_date(trim(DATADENASCIMENTO), 'dd/MM/yyyy')
            end as DATADENASCIMENTO,
            
            try_cast(var_03 as int) as var_03,
            try_cast(var_02 as int) as var_02,
            try_cast(var_04 as int) as var_04,
            try_cast(var_05 as int) as var_05,
            try_cast(var_06 as int) as var_06,
            try_cast(var_07 as int) as var_07,
            try_cast(var_08 as int) as var_08,
            try_cast(var_09 as int) as var_09,
            try_cast(var_10 as int) as var_10,
            try_cast(var_11 as int) as var_11,
            
            -- Datas com tratamento para valores nulos/vazios usando to_date
            case 
                when trim(var_12) in ('null', 'NULL', '') then null
                else try_to_date(trim(var_12), 'dd/MM/yyyy')
            end as var_12,
            
            case 
                when trim(var_13) in ('null', 'NULL', '') then null
                else try_to_date(trim(var_13), 'dd/MM/yyyy')
            end as var_13,
            
            try_cast(var_14 as INT) as var_14,
            try_cast(var_15 as STRING) as var_15,
            try_cast(var_16 as INT) as var_16,
            try_cast(var_17 as INT) as var_17,
            try_cast(var_18 as STRING) as var_18,
            try_cast(var_19 as STRING) as var_19,
            try_cast(var_20 as INT) as var_20,
            try_cast(var_21 as STRING) as var_21,
            try_cast(var_22 as STRING) as var_22,
            try_cast(var_23 as STRING) as var_23,
            try_cast(var_24 as STRING) as var_24,
            try_cast(var_25 as STRING) as var_25,
            try_cast(CEP_3_digitos as int) as CEP_3_digitos,
            
            {pdthproc} as DATPROC

        from
            raw_00
            
    """.format(pdthproc=dthproc))
lake.createOrReplaceTempView("lake")





lake_dedup = spark.sql("""
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY NUM_CPF, SAFRA
                ORDER BY DATPROC DESC
            ) AS rn
        FROM lake
    ) t
    WHERE rn = 1
""")
lake_dedup.createOrReplaceTempView("lake_dedup")



from delta.tables import DeltaTable

silver_table = "hackathon2025.silver.base_dados_cadastrais"



(
        lake_dedup
        .write
        .format("delta")
        .mode("overwrite")
        .partitionBy("SAFRA")
        .option("mergeSchema", "true")
        .saveAsTable(silver_table) 
)

print("Tabela silver criada com sucesso")





name = "base_dados_cadastrais"

df_controle = spark.sql("""
    SELECT
        '{name_table}'        AS nome_tabela,
        SAFRA                 AS safra,
        COUNT(*)              AS qtd_registros,
        current_timestamp()   AS datproc
    FROM lake_dedup
    GROUP BY SAFRA
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