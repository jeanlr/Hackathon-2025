#  Hackathon2025 - Projeto Squad 07 - Data Lake - Machine Learning

## Como subir o projeto?
```bash
git clone https://github.com/jeanlr/Hackathon-2025
cd Hackathon-2025
# No terminal, execute os comandos abaixo: (Ajusta o dono das pastas para o usuário interno do Airflow (UID 50000))
sudo chown -R 50000:0 dags logs plugins scripts
# Garante permissão total de escrita:
sudo chmod -R 777 .

## adicionar arquivo .env na raiz do projeto:
# Navegue até a raiz do seu projeto (se ainda não estiver lá)
# cd /caminho/para/seu/projeto

# Cria o arquivo .env
touch .env

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

# Criar uma pasta chamada "silver"
# Na pasta silver adicionar os notebooks:
  1. base_telco
  2. base_cadastral
```

## Airflow
```bash
# Entrar na url:
http://localhost:8080

```

## Spark UI
```bash
# Entrar na url:
http://localhost:8085

```
