from pyspark.sql import SparkSession
from datetime import datetime
import pytz
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import datetime
from delta import *




agora=datetime.now(pytz.timezone('America/Sao_Paulo'))
dthproc=agora.strftime("%Y%m%d%H%M%S")



path = "hackathon2025.silver.base_score_bureau_movel"
df_base_score_bureau_movel = spark.read.table(path)





path = "hackathon2025.silver.base_telco"
df_base_telco = spark.read.table(path)



path = "hackathon2025.silver.base_dados_cadastrais"
df_base_dados_cadastrais = spark.read.table(path)



df_base_score_bureau_movel.createOrReplaceTempView("bureau")
df_base_telco.createOrReplaceTempView("telco")
df_base_dados_cadastrais.createOrReplaceTempView("cadastral")




score_cols = set(df_base_score_bureau_movel.columns)
telco_cols = set(df_base_telco.columns) - score_cols
cad_cols   = set(df_base_dados_cadastrais.columns) - score_cols




telco_select = ",\n    ".join([f"t.{c}" for c in telco_cols])
cad_select   = ",\n    ".join([f"c.{c}" for c in cad_cols])

query = f"""
SELECT
    b.*,
    {telco_select},
    {cad_select}
FROM bureau b
LEFT JOIN telco t
    ON b.NUM_CPF = t.NUM_CPF AND b.SAFRA  = t.SAFRA 
LEFT JOIN cadastral c
    ON b.NUM_CPF = c.NUM_CPF AND b.SAFRA  = c.SAFRA 
"""
df_final = spark.sql(query)




df_final.createOrReplaceTempView("df_final")



one_big_table = spark.sql(     
    """
        select
        
            -- campos do arquivo --

            NUM_CPF,
            SAFRA,
            FLAG_INSTALACAO,
            FPD,
            PROD,
            flag_mig2,
            SCORE_01,
            SCORE_02,       
            var_37,
            var_28,
            var_92,
            var_51,
            var_50,
            var_81,
            var_48,
            var_88,
            var_46,
            var_27,
            var_70,
            var_87,
            var_26,
            var_40,
            var_30,
            var_62,
            var_36,
            var_56,
            var_39,
            var_76,
            var_49,
            var_38,
            var_68,
            var_52,
            var_89,
            var_64,
            var_33,
            var_82,
            var_66,
            var_69,
            var_84,
            var_61,
            var_67,
            var_43,
            var_32,
            var_75,
            var_65,
            var_83,
            var_78,
            var_47,
            var_79,
            var_44,
            var_90,
            var_57,
            var_58,
            var_35,
            var_42,
            var_29,
            var_72,
            var_31,
            var_73,
            var_53,
            var_34,
            var_71,
            var_80,
            var_63,
            var_86,
            var_85,
            var_93,
            var_54,
            var_45,
            var_74,
            var_55,
            var_41,
            var_77,
            var_91,
            var_59,
            var_60,
            var_17,
            var_11,
            var_04,
            var_22,
            CEP_3_digitos,
            var_23,
            var_09,
            case 
                when trim(DATADENASCIMENTO) in ('null', 'NULL', '', '-3', '-2', '-1') then null
                else cast(
                    cast(SAFRA / 100 as int) 
                    - year(to_date(trim(DATADENASCIMENTO), 'yyyy-MM-dd'))
                    as int
                )
            end as IDADE,
            var_15,
            var_25,
            var_14,
            var_02,
            var_18,
            var_20,
            var_21,
            var_16,
            var_19,
            var_13,
            var_03,
            var_12,
            var_07,
            var_06,
            var_05,
            var_24,
            var_10,
            var_08,
            STATUSRF,
            {pdthproc} as DATPROC

        from
            df_final
        order by
            NUM_CPF,
            SAFRA
            
    """.format(pdthproc=dthproc))
one_big_table.createOrReplaceTempView("one_big_table")
o


# Deduplicação caso aconteça
one_big_table_dedup = one_big_table.dropDuplicates()


one_big_table_dedup.createOrReplaceTempView("one_big_table_dedup")




gold_table = "hackathon2025.gold.one_big_table_publico"



(
        one_big_table_dedup
        .write
        .format("delta")
        .mode("overwrite")
        .partitionBy("SAFRA")
        .option("mergeSchema", "true")
        .saveAsTable(gold_table) 
)

print("Tabela gold criada com sucesso")






name = "one_big_table_publico"

df_controle = spark.sql("""
    SELECT
        '{name_table}'        AS nome_tabela,
        SAFRA                 AS safra,
        COUNT(*)              AS qtd_registros,
        current_timestamp()   AS datproc
    FROM one_big_table_dedup
    GROUP BY SAFRA
    ORDER BY SAFRA
""".format(name_table=name))




gold_table_controle = "hackathon2025.gold.controle"
if not spark.catalog.tableExists(gold_table_controle):
    print("Tabela gold_controle não existe. Criando...")

    (
        df_controle
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(gold_table_controle)  
    )

    print("Tabela gold_controle criada com sucesso")

else:
    print("Tabela de controle existe. Inserindo novo registro...")

    delta_silver = DeltaTable.forName(spark, gold_table_controle)

    (
        df_controle
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(gold_table_controle)
    )
    print("Dados inseridos com sucesso...")    