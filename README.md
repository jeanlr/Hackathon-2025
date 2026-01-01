#  Hackathon2025 - Projeto Squad 07 - Data Lake - Machine Learning

## Arquitetura do projeto
![bucket_bronze](/assets/arquitetura.jpg)
---

### Visão geral da arquitetura

A arquitetura proposta implementa um Data Lakehouse moderno, containerizado e desacoplado, projetado para suportar ingestão, processamento e consumo de dados de forma confiável, segura e escalável.

A solução é baseada em Docker Compose, permitindo reprodutibilidade do ambiente e isolamento de serviços.

Essa abordagem atende tanto demandas analíticas quanto operacionais, suportando desde cargas batch até exploração avançada e ciência de dados.

### Justificativa das escolhas

- **MinIO (Object Storage – S3 Compatible)**  
  Escolhido por ser escalável, performático e aderente ao padrão S3, evitando lock-in tecnológico e permitindo migração futura para cloud pública sem alterações no código.

- **Apache Spark**
  Proporciona processamento distribuído e escalável para grandes volumes de dados, sendo adequado para ELT/ETL, feature engineering e cargas analíticas complexas.

- **Apache Airflow** 
  Responsável pela orquestração, governança e observabilidade dos pipelines, garantindo reprocessamentos seguros, monitoramento e maior confiabilidade operacional.

- **Jupyter Notebook**  
  Oferece um ambiente flexível para exploração, validação e análises avançadas, integrado ao Spark, acelerando o ciclo analítico e reduzindo o tempo de experimentação.

---

## Como subir o projeto?
```bash
git clone https://github.com/jeanlr/Hackathon-2025
cd Hackathon-2025
git fetch --all --prune
## exemplo, local já configura a branch de trabalho sincronizado com o remoto.
git switch -c engenharia-de-dados origin/engenharia-de-dados

# No terminal, execute os comandos abaixo: (Ajusta o dono das pastas para o usuário interno do Airflow (UID 50000))
sudo chown -R 50000:0 dags logs plugins scripts
# Garante permissão total de escrita:
sudo chmod -R 777 .

## adicionar arquivo .env na raiz do projeto:
# Navegue até a raiz do seu projeto (se ainda não estiver lá)
# cd /caminho/para/seu/projeto

# Cria o arquivo .env
touch .env
Utilizar o arquivo .env.example para as suas credenciais

# Adiciona .env ao .gitignore para que o Git o ignore
echo ".env" >> .gitignore

!Lembre de adicionar as credenciais no arquivo .env

# Subir containers Docker:
docker compose up -d

```

## Subir arquivos para o bucket bronze dentro do storage MinIO (port=9001 local)

![bucket_bronze](/assets/bucket_bronze.png)

## Ambiente Jupyter

```bash
# Entrar na url:
http://127.0.0.1:8888/lab


# Se não aparecer as pastas jupyter-notebooks, Criar uma pasta chamada "jupyter-notebooks".
# Na pasta, criar uma outra chamada "silver" e adicionar os notebooks:
  1. base_telco
  2. base_cadastral
  3. base_bureau
  4. base_pagamento
  5. base_atraso
  6. base_recarga

# Na pasta, criar uma outra chamada "gold" e adicionar os notebooks:
  1. one_big_table_publico
```

## Airflow
```bash
# Entrar na url:
http://localhost:8080

# Na pasta dags há os scripts .py que processando os dados.

```

## Spark UI
```bash
# Entrar na url:
http://localhost:8085

```
