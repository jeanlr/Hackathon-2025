from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz


agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")


df_book_atraso = spark.read.parquet("/Volumes/hackathon2025/bronze/book_atraso/")



df_book_atraso.createOrReplaceTempView("raw_00")




raw_00_com_safra = spark.sql("""
    SELECT
        *,
        CAST(
            date_format(
                to_timestamp(DAT_REFERENCIA, 'ddMMMyyyy:HH:mm:ss'),
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
            case 
                when trim(DAT_REFERENCIA) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_REFERENCIA), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_REFERENCIA,
            try_cast(NUM_FATURA_HASH as STRING) as NUM_FATURA_HASH,
            try_cast(NUM_ENT_SEQ_FATURA as INT) as NUM_ENT_SEQ_FATURA,
            try_cast(CONTRATO as BIGINT) as CONTRATO,
            try_cast(DW_UN_NEGOCIO as INT) as DW_UN_NEGOCIO,
            try_cast(DW_HIS_PONTO_VENDA_COMTA as BIGINT) as DW_HIS_PONTO_VENDA_COMTA,
            try_cast(DW_NUM_CLIENTE as STRING) as DW_NUM_CLIENTE,
            try_cast(DW_AREA as INT) as DW_AREA,
            try_cast(DW_CICLO as INT) as DW_CICLO,
            try_cast(DW_TIPO_CLIENTE_CONTA as STRING) as DW_TIPO_CLIENTE_CONTA,
            try_cast(DW_OFERTA as STRING) as DW_OFERTA,
            try_cast(DW_FAIXA_AGING_FATURA as INT) as DW_FAIXA_AGING_FATURA,
            try_cast(DW_FAIXA_AGING_DIVIDA as INT) as DW_FAIXA_AGING_DIVIDA,
            try_cast(DW_FAIXA_TEMPO_BASE as INT) as DW_FAIXA_TEMPO_BASE,
            try_cast(DW_FAIXA_AGING_PROX_FECH as INT) as DW_FAIXA_AGING_PROX_FECH,
            try_cast(DW_TIPO_FATURAMENTO as STRING) as DW_TIPO_FATURAMENTO,
            try_cast(COD_PLATAFORMA as STRING) as COD_PLATAFORMA,
            case 
                when trim(DAT_CRIACAO_REGISTRO_TRANS) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_REGISTRO_TRANS), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_REGISTRO_TRANS,
            case 
                when trim(DAT_ALTERACAO_REGISTRO_TRANS) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_ALTERACAO_REGISTRO_TRANS), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ALTERACAO_REGISTRO_TRANS,
            case 
                when trim(DAT_CANCELAMENTO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_CANCELAMENTO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CANCELAMENTO_FAT,
            case 
                when trim(DAT_ORIGINAL_VCTO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_ORIGINAL_VCTO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ORIGINAL_VCTO_FAT,
            case 
                when trim(DAT_ALTERACAO_VCTO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_ALTERACAO_VCTO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ALTERACAO_VCTO_FAT,
            case 
                when trim(DAT_CRIACAO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_FAT,
            case 
                when trim(DAT_VENCIMENTO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_VENCIMENTO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_VENCIMENTO_FAT,
            case 
                when trim(DAT_STATUS_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_STATUS_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_STATUS_FAT,
            case 
                when trim(DAT_MIN_VENCIMENTO_FAT) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_MIN_VENCIMENTO_FAT), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_MIN_VENCIMENTO_FAT,
            try_cast(NUM_BILL_SEQ_FAT as INT) as NUM_BILL_SEQ_FAT,
            try_cast(NUM_SEQ_ACORDO_FAT as INT) as NUM_SEQ_ACORDO_FAT,
            try_cast(IND_ISENCAO_COB_FAT as STRING) as IND_ISENCAO_COB_FAT,
            try_cast(IND_WO as STRING) as IND_WO,
            try_cast(IND_PDD as STRING) as IND_PDD,
            try_cast(IND_PCCR as STRING) as IND_PCCR,
            try_cast(IND_ACA as STRING) as IND_ACA,
            try_cast(IND_PRIMEIRA_FAT as STRING) as IND_PRIMEIRA_FAT,
            try_cast(IND_FRAUDE as STRING) as IND_FRAUDE,
            try_cast(VAL_FAT_LIQUIDO as DECIMAL(10,2)) as VAL_FAT_LIQUIDO,
            try_cast(VAL_FAT_BRUTO as DECIMAL(10,2)) as VAL_FAT_BRUTO,
            try_cast(VAL_FAT_CREDITO as DECIMAL(10,2)) as VAL_FAT_CREDITO,
            try_cast(VAL_FAT_AJUSTE as DECIMAL(10,2)) as VAL_FAT_AJUSTE,
            try_cast(VAL_FAT_BRUTO_BC as DECIMAL(10,2)) as VAL_FAT_BRUTO_BC,
            try_cast(VAL_FAT_PAGAMENTO_BRUTO as DECIMAL(10,2)) as VAL_FAT_PAGAMENTO_BRUTO,
            try_cast(VAL_FAT_ABERTO as DECIMAL(10,2)) as VAL_FAT_ABERTO,
            try_cast(VAL_FAT_ABERTO_LIQ as DECIMAL(10,2)) as VAL_FAT_ABERTO_LIQ,
            try_cast(VAL_MULTA_JUROS as DECIMAL(10,2)) as VAL_MULTA_JUROS,
            try_cast(VAL_MULTA_CANCELAMENTO as DECIMAL(10,2)) as VAL_MULTA_CANCELAMENTO,
            try_cast(VAL_PARC_APARELHO_LIQ as DECIMAL(10,2)) as VAL_PARC_APARELHO_LIQ,
            try_cast(VAL_FAT_LIQ_JM_MC as DECIMAL(10,2)) as VAL_FAT_LIQ_JM_MC,
            case 
                when trim(DAT_ATIVACAO_CONTA_CLI) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_ATIVACAO_CONTA_CLI), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ATIVACAO_CONTA_CLI,
            case 
                when trim(DAT_CRIACAO_DW) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_DW), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_DW,
            {pdthproc} as DATPROC

        from
            raw_00_com_safra
            
    """.format(pdthproc=dthproc))
lake.createOrReplaceTempView("lake")



# Deduplicação caso aconteça de reprocessar mesma base
lake_dedup = spark.sql("""
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY NUM_CPF, DAT_REFERENCIA, CONTRATO, NUM_ENT_SEQ_FATURA
                ORDER BY DATPROC DESC
            ) AS rn
        FROM lake
    ) t
    WHERE rn = 1
""")
lake_dedup.createOrReplaceTempView("lake_dedup")


from delta.tables import DeltaTable

silver_table = "hackathon2025.silver.base_atraso"


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



name = "base_atraso"

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