from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
import pytz





agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")



df_book_pagamento = spark.read.parquet("/Volumes/hackathon2025/bronze/book_pagamento/")




df_book_pagamento.createOrReplaceTempView("raw_00")





raw_00_com_safra = spark.sql("""
    SELECT
        *,
        CAST(
            date_format(
                to_timestamp(DAT_STATUS_FATURA, 'ddMMMyyyy:HH:mm:ss'),
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
                when trim(DAT_STATUS_FATURA) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_STATUS_FATURA), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_STATUS_FATURA,
            try_cast(CONTRATO as BIGINT) as CONTRATO,
            try_cast(SEQ_FATURA as INT) as SEQ_FATURA,
            try_cast(NUM_SUB_SEQ_FATURA as INT) as NUM_SUB_SEQ_FATURA,
            try_cast(NUM_CREDITO_SEQ as INT) as NUM_CREDITO_SEQ,
            try_cast(DW_TIPO_FATURA as STRING) as DW_TIPO_FATURA,
            try_cast(IND_STATUS_FATURA as STRING) as IND_STATUS_FATURA,
            try_cast(DW_NUM_CLIENTE as STRING) as DW_NUM_CLIENTE,
            try_cast(DW_AREA as INT) as DW_AREA,
            try_cast(DW_UN_NEGOCIO as STRING) as DW_UN_NEGOCIO,
            try_cast(DW_FORMA_PAGAMENTO as STRING) as DW_FORMA_PAGAMENTO,
            try_cast(VAL_PAGAMENTO_FATURA as DECIMAL(10,2)) as VAL_PAGAMENTO_FATURA,
            case 
                when trim(DAT_CRIACAO_DW) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_DW), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_DW,
            try_cast(DW_BANCO as STRING) as DW_BANCO,
            try_cast(DW_TIPO_PAGAMENTO as STRING) as DW_TIPO_PAGAMENTO,
            try_cast(NUM_BANCO_PAGAMENTO as STRING) as NUM_BANCO_PAGAMENTO,
            try_cast(NUM_AGENCIA_PAGAMENTO as STRING) as NUM_AGENCIA_PAGAMENTO,
            try_cast(NUM_CC_PAGAMENTO as STRING) as NUM_CC_PAGAMENTO,
            try_cast(DW_MOTIVO_ESTORNO as STRING) as DW_MOTIVO_ESTORNO,
            try_cast(VAL_DESCONTO_ITEM as DECIMAL(10,2)) as VAL_DESCONTO_ITEM,
            try_cast(VAL_PAGAMENTO_ITEM as DECIMAL(10,2)) as VAL_PAGAMENTO_ITEM,
            try_cast(VAL_JUROS_MULTAS_ITEM as DECIMAL(10,2)) as VAL_JUROS_MULTAS_ITEM,
            try_cast(VAL_MULTA_EQUIP_ITEM as DECIMAL(10,2)) as VAL_MULTA_EQUIP_ITEM,
            try_cast(VAL_MULTA_EQUIP_TOTAL as DECIMAL(10,2)) as VAL_MULTA_EQUIP_TOTAL,
            try_cast(VAL_MULTA_FID_ITEM as DECIMAL(10,2)) as VAL_MULTA_FID_ITEM,
            try_cast(COD_ORIGEM_NETUNO as STRING) as COD_ORIGEM_NETUNO,
            try_cast(COD_CONTA_ATIVIDADE as STRING) as COD_CONTA_ATIVIDADE,
            try_cast(SEQ_ENTIDADE_ATIVIDADE as INT) as SEQ_ENTIDADE_ATIVIDADE,
            case 
                when trim(DAT_CRIACAO_ATIVIDADE) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_ATIVIDADE), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_ATIVIDADE,
            case 
                when trim(DAT_ATUALIZACAO_ATIVIDADE) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_ATUALIZACAO_ATIVIDADE), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ATUALIZACAO_ATIVIDADE,
            try_cast(COD_LOGIN_OPERADOR_ATIVIDADE as STRING) as COD_LOGIN_OPERADOR_ATIVIDADE,
            try_cast(COD_ATIVIDADE as STRING) as COD_ATIVIDADE,
            try_cast(COD_RAZAO_ATIVIDADE as STRING) as COD_RAZAO_ATIVIDADE,
            case 
                when trim(DAT_BAIXA_ATIVIDADE) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_BAIXA_ATIVIDADE), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_BAIXA_ATIVIDADE,
            try_cast(VAL_BAIXA_ATIVIDADE as DECIMAL(10,2)) as VAL_BAIXA_ATIVIDADE,
            case 
                when trim(DAT_DEPOSITO_ATIVIDADE) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_DEPOSITO_ATIVIDADE), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_DEPOSITO_ATIVIDADE,
            try_cast(COD_FUNDO_ATIVIDADE as STRING) as COD_FUNDO_ATIVIDADE,
            try_cast(COD_BANCO_ATIVIDADE as STRING) as COD_BANCO_ATIVIDADE,
            try_cast(NUM_CONTA_ATIVIDADE as STRING) as NUM_CONTA_ATIVIDADE,
            try_cast(COD_AGENCIA_ATIVIDADE as STRING) as COD_AGENCIA_ATIVIDADE,
            try_cast(SEQ_ENTIDADE_PAGAMENTO as INT) as SEQ_ENTIDADE_PAGAMENTO,
            case 
                when trim(DAT_CRIACAO_PAGAMENTO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_PAGAMENTO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_PAGAMENTO,
            case 
                when trim(DAT_ATUALIZACAO_PAGAMENTO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_ATUALIZACAO_PAGAMENTO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ATUALIZACAO_PAGAMENTO,
            try_cast(COD_LOGIN_PAGAMENTO as STRING) as COD_LOGIN_PAGAMENTO,
            try_cast(COD_FORMA_PAGAMENTO as STRING) as COD_FORMA_PAGAMENTO,
            try_cast(VAL_ORIGINAL_PAGAMENTO as DECIMAL(10,2)) as VAL_ORIGINAL_PAGAMENTO,
            try_cast(NUM_FATURA_PAGAMENTO as STRING) as NUM_FATURA_PAGAMENTO,
            try_cast(COD_TIPO_PAGAMENTO as STRING) as COD_TIPO_PAGAMENTO,
            try_cast(DSC_NOME_BANCO_PAGAMENTO as STRING) as DSC_NOME_BANCO_PAGAMENTO,
            try_cast(SEQ_ARQUIVO_PAGAMENTO as INT) as SEQ_ARQUIVO_PAGAMENTO,
            try_cast(NUM_PARCELA_PAGAMENTO as STRING) as NUM_PARCELA_PAGAMENTO,
            try_cast(NUM_AGRUPADOR_PAGAMENTO as STRING) as NUM_AGRUPADOR_PAGAMENTO,
            try_cast(DSC_PAGAMENTO as STRING) as DSC_PAGAMENTO,
            try_cast(VAL_ATUAL_PAGAMENTO as DECIMAL(10,2)) as VAL_ATUAL_PAGAMENTO,
            try_cast(COD_METODO_PAGAMENTO as INT) as COD_METODO_PAGAMENTO,
            try_cast(IND_STATUS_PAGAMENTO as STRING) as IND_STATUS_PAGAMENTO,
            case 
                when trim(DAT_STATUS_PAGAMENTO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_STATUS_PAGAMENTO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_STATUS_PAGAMENTO,
            try_cast(COD_ARQUIVO_PAGAMENTO as STRING) as COD_ARQUIVO_PAGAMENTO,
            try_cast(COD_NETUNO_PAGAMENTO as STRING) as COD_NETUNO_PAGAMENTO,
            case 
                when trim(DAT_CRIACAO_CREDITO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_CRIACAO_CREDITO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_CRIACAO_CREDITO,
            case 
                when trim(DAT_ATUALIZACAO_CREDITO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_ATUALIZACAO_CREDITO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ATUALIZACAO_CREDITO,
            try_cast(COD_LOGIN_CREDITO as STRING) as COD_LOGIN_CREDITO,
            try_cast(VAL_PAGAMENTO_CREDITO as DECIMAL(10,2)) as VAL_PAGAMENTO_CREDITO,
            try_cast(IND_TIPO_CREDITO as STRING) as IND_TIPO_CREDITO,
            try_cast(SEQ_PAGAMENTO_CREDITO as INT) as SEQ_PAGAMENTO_CREDITO,
            try_cast(SEQ_FATURA_CREDITO as INT) as SEQ_FATURA_CREDITO,
            try_cast(COD_ALOCACAO_CREDITO as STRING) as COD_ALOCACAO_CREDITO,
            try_cast(COD_DESALOCACAO_CREDITO as STRING) as COD_DESALOCACAO_CREDITO,
            try_cast(SEQ_ENTIDADE_CREDITO as INT) as SEQ_ENTIDADE_CREDITO,
            try_cast(COD_TIPO_FATURA as STRING) as COD_TIPO_FATURA,
            case 
                when trim(DAT_ATIVIDADE_CREDITO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_ATIVIDADE_CREDITO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_ATIVIDADE_CREDITO,
            case 
                when trim(DAT_VENCIMENTO_CREDITO) in ('null', 'NULL', '') then null
                else try_cast(to_timestamp(trim(DAT_VENCIMENTO_CREDITO), 'ddMMMyyyy:HH:mm:ss') as TIMESTAMP)
            end as DAT_VENCIMENTO_CREDITO,
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
                PARTITION BY NUM_CPF, DAT_STATUS_FATURA, CONTRATO, NUM_SUB_SEQ_FATURA
                ORDER BY DATPROC DESC
            ) AS rn
        FROM lake
    ) t
    WHERE rn = 1
""")
lake_dedup.createOrReplaceTempView("lake_dedup")



from delta.tables import DeltaTable

silver_table = "hackathon2025.silver.base_pagamento"


(
        lake_dedup
        .write
        .format("delta")
        .mode("overwrite")
        .partitionBy("SAFRA")
        .option("overwriteSchema", "true")
        .saveAsTable(silver_table) 
)

print("Safra criada com sucesso")





name = "base_pagamento"

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