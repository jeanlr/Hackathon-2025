from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz



agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")


df_base_score_bureau_movel = spark.read.parquet("/Volumes/hackathon2025/bronze/base_score_bureau_movel/")



df_base_score_bureau_movel.createOrReplaceTempView("raw_00")



lake = spark.sql(     
    """
        select
        
            -- campos do arquivo --

            cast(NUM_CPF as string) as NUM_CPF,
            cast(SAFRA as int) as SAFRA,
            cast(FLAG_INSTALACAO as int) as FLAG_INSTALACAO,
            cast(FPD as int) as FPD,
            cast(PROD as string) as PROD,
            cast(flag_mig2 as string) as flag_mig2,
            cast(SCORE_01 as int) as SCORE_01,
            cast(SCORE_02 as int) as SCORE_02,
            {pdthproc} as DATPROC

        from
            raw_00
            
    """.format(pdthproc=dthproc))
lake.createOrReplaceTempView("lake")




for col in lake.columns:
    agg_result = lake.agg(
        {col: "count"} 
    ).collect()[0]
    
    total = lake.count()
    nao_nulos = agg_result[f"count({col})"]
    nulos = total - nao_nulos
    
    if nulos > 0:
        print(f"{col}: {nulos} nulos ({nulos/total*100:.2f}%)")


# Deduplicação caso aconteça de reprocessar mesma base
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

silver_table = "hackathon2025.silver.base_score_bureau_movel"



(
        lake_dedup
        .write
        .format("delta")
        .mode("overwrite")
        .partitionBy("SAFRA")
        .option("mergeSchema", "true")
        .saveAsTable(silver_table)
)    

print("Safra criada com sucesso")






name = "base_score_bureau_movel"

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