from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz





agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")



df_book_recarga = spark.read.parquet("/Volumes/hackathon2025/bronze/book_recarga/")




df_book_recarga.createOrReplaceTempView("raw_00")




raw_00_com_safra = spark.sql("""
    SELECT
        *,
        CAST(
            date_format(
                to_timestamp(DAT_INSERCAO_CREDITO, 'ddMMMyyyy:HH:mm:ss'),
                'yyyyMM'
            ) AS INT
        ) AS SAFRA
    FROM raw_00
""")

raw_00_com_safra.createOrReplaceTempView("raw_00_com_safra")





lake = spark.sql(     
    """
        select
        
            -- campos do arquivo --

            try_cast(NUM_CPF as STRING) as NUM_CPF,
            try_cast(SAFRA as INT) as SAFRA,
            try_cast(DW_NUM_NTC as STRING) as DW_NUM_NTC,
            case 
                when trim(DAT_INSERCAO_CREDITO) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_INSERCAO_CREDITO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_INSERCAO_CREDITO,
            case 
                when trim(HOR_INSERCAO_CREDITO) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else
                    cast(substr(lpad(trim(HOR_INSERCAO_CREDITO), 6, '0'), 1, 2) as int) * 3600 +
                    cast(substr(lpad(trim(HOR_INSERCAO_CREDITO), 6, '0'), 3, 2) as int) * 60 +
                    cast(substr(lpad(trim(HOR_INSERCAO_CREDITO), 6, '0'), 5, 2) as int)
            end as HOR_INSERCAO_CREDITO,
            try_cast(DW_NUM_CLIENTE as STRING) as DW_NUM_CLIENTE,
            try_cast(COD_TECNOLOGIA_DW as STRING) as COD_TECNOLOGIA_DW,
            try_cast(COD_CANAL_AQUISICAO as STRING) as COD_CANAL_AQUISICAO,
            try_cast(COD_TIPO_CREDITO as STRING) as COD_TIPO_CREDITO,
            try_cast(COD_PROMOCAO as STRING) as COD_PROMOCAO,
            try_cast(VAL_CREDITO_INSERIDO as DECIMAL(10,2)) as VAL_CREDITO_INSERIDO,
            try_cast(VAL_BONUS as DECIMAL(10,2)) as VAL_BONUS,
            try_cast(VAL_REAL as DECIMAL(10,2)) as VAL_REAL,
            try_cast(COD_PLATAFORMA_ATU as STRING) as COD_PLATAFORMA_ATU,
            try_cast(COD_STATUS_PLATAFORMA as STRING) as COD_STATUS_PLATAFORMA,
            try_cast(IND_METODO_PAGAMENTO as STRING) as IND_METODO_PAGAMENTO,
            try_cast(DW_PLANO_TARIFACAO as STRING) as DW_PLANO_TARIFACAO,
            try_cast(DW_TIPO_RECARGA as STRING) as DW_TIPO_RECARGA,
            try_cast(DW_TIPO_INSERCAO as STRING) as DW_TIPO_INSERCAO,
            try_cast(DW_FORMA_PAGAMENTO as STRING) as DW_FORMA_PAGAMENTO,
            try_cast(DW_INSTITUICAO as STRING) as DW_INSTITUICAO,
            try_cast(COD_GRUPO_CARTAO as STRING) as COD_GRUPO_CARTAO,
            try_cast(DSC_GRUPO_CARTAO_WPP as STRING) as DSC_GRUPO_CARTAO_WPP,
            try_cast(FLAG_SOS as INT) as FLAG_SOS,
            try_cast(VALOR_SOS as INT) as VALOR_SOS,
            {pdthproc} as DATPROC

        from
            raw_00_com_safra
            
    """.format(pdthproc=dthproc))

lake.createOrReplaceTempView("lake")



lake_exemplo_dedup = spark.sql("""
    SELECT *
    FROM lake
    WHERE NUM_CPF = "79TUUYWXYWU" AND DW_NUM_NTC = 578905048 AND DAT_INSERCAO_CREDITO = "2023-10-06 00:00:00"
    ORDER BY DW_NUM_NTC, DAT_INSERCAO_CREDITO, HOR_INSERCAO_CREDITO
""")



# Deduplicação caso aconteça de reprocessar mesma base
lake_dedup = lake.dropDuplicates()



lake_dedup.createOrReplaceTempView("lake_dedup")



from delta.tables import DeltaTable

silver_table = "hackathon2025.silver.base_recarga"



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




name = "base_recarga"

df_controle = spark.sql("""
    SELECT
        '{name_table}'        AS nome_tabela,
        SAFRA                 AS safra,
        COUNT(*)              AS qtd_registros,
        current_timestamp()   AS datproc
    FROM lake_dedup
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