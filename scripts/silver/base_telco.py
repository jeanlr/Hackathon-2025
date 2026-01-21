from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz



agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")


df_base_telco = spark.read.parquet("/Volumes/hackathon2025/bronze/base_telco/")




df_base_telco.createOrReplaceTempView("raw_00")




lake = spark.sql(     
    """
        select
        
            -- colunas do arquivo --

            try_cast(NUM_CPF as STRING) as NUM_CPF,
            try_cast(SAFRA as INT) as SAFRA,
            try_cast(FLAG_INSTALACAO as INT) as FLAG_INSTALACAO,
            try_cast(FPD as INT) as FPD,
            try_cast(PROD as STRING) as PROD,
            try_cast(flag_mig2 as STRING) as flag_mig2,
            try_cast(var_26 as DOUBLE) as var_26,
            try_cast(var_27 as DOUBLE) as var_27,
            try_cast(var_28 as DOUBLE) as var_28,
            try_cast(var_29 as DOUBLE) as var_29,
            try_cast(var_30 as DOUBLE) as var_30,
            try_cast(var_31 as DOUBLE) as var_31,
            try_cast(var_32 as DOUBLE) as var_32,
            try_cast(var_33 as DOUBLE) as var_33,
            try_cast(var_34 as DOUBLE) as var_34,
            try_cast(var_35 as DOUBLE) as var_35,
            try_cast(var_36 as DOUBLE) as var_36,
            try_cast(var_37 as DOUBLE) as var_37,
            try_cast(var_38 as DOUBLE) as var_38,
            try_cast(var_39 as DOUBLE) as var_39,
            try_cast(var_40 as DOUBLE) as var_40,
            try_cast(var_41 as DOUBLE) as var_41,
            try_cast(var_42 as DOUBLE) as var_42,
            try_cast(var_43 as DOUBLE) as var_43,
            try_cast(var_44 as DOUBLE) as var_44,
            try_cast(var_45 as DOUBLE) as var_45,
            try_cast(var_46 as DOUBLE) as var_46,
            try_cast(var_47 as DOUBLE) as var_47,
            try_cast(var_48 as DOUBLE) as var_48,
            try_cast(var_49 as DOUBLE) as var_49,
            try_cast(var_50 as DOUBLE) as var_50,
            try_cast(var_51 as DOUBLE) as var_51,
            try_cast(var_52 as DOUBLE) as var_52,
            try_cast(var_53 as DOUBLE) as var_53,
            try_cast(var_54 as DOUBLE) as var_54,
            try_cast(var_55 as DOUBLE) as var_55,
            try_cast(var_56 as DOUBLE) as var_56,
            try_cast(var_57 as DOUBLE) as var_57,
            try_cast(var_58 as DOUBLE) as var_58,
            try_cast(var_59 as DOUBLE) as var_59,
            try_cast(var_60 as DOUBLE) as var_60,
            try_cast(var_61 as DOUBLE) as var_61,
            try_cast(var_62 as DOUBLE) as var_62,
            try_cast(var_63 as DOUBLE) as var_63,
            try_cast(var_64 as DOUBLE) as var_64,
            try_cast(var_65 as DOUBLE) as var_65,
            try_cast(var_66 as DOUBLE) as var_66,
            try_cast(var_67 as DOUBLE) as var_67,
            try_cast(var_68 as DOUBLE) as var_68,
            try_cast(var_69 as DOUBLE) as var_69,
            try_cast(var_70 as DOUBLE) as var_70,
            try_cast(var_71 as DOUBLE) as var_71,
            try_cast(var_72 as DOUBLE) as var_72,
            try_cast(var_73 as DOUBLE) as var_73,
            try_cast(var_74 as DOUBLE) as var_74,
            try_cast(var_75 as DOUBLE) as var_75,
            try_cast(var_76 as DOUBLE) as var_76,
            try_cast(var_77 as DOUBLE) as var_77,
            try_cast(var_78 as DOUBLE) as var_78,
            try_cast(var_79 as DOUBLE) as var_79,
            try_cast(var_80 as DOUBLE) as var_80,
            try_cast(var_81 as DOUBLE) as var_81,
            try_cast(var_82 as DOUBLE) as var_82,
            try_cast(var_83 as DOUBLE) as var_83,
            try_cast(var_84 as DOUBLE) as var_84,
            try_cast(var_85 as DOUBLE) as var_85,
            try_cast(var_86 as DOUBLE) as var_86,
            try_cast(var_87 as DOUBLE) as var_87,
            try_cast(var_88 as DOUBLE) as var_88,
            try_cast(var_89 as DOUBLE) as var_89,
            try_cast(var_90 as DOUBLE) as var_90,
            try_cast(var_91 as DOUBLE) as var_91,
            try_cast(var_92 as DOUBLE) as var_92,
            try_cast(var_93 as DOUBLE) as var_93,
            {pdthproc} as DATPROC

        from
            raw_00
            
    """.format(pdthproc=dthproc))
lake.createOrReplaceTempView("lake")


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

silver_table = "hackathon2025.silver.base_telco"



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




name = "base_telco"

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
        .saveAsTable(silver_table_controle)   # 👈 salva direto no schema silver
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