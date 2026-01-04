#  Hackathon2025 - Projeto Squad 07 - Data Lake - Machine Learning

## Arquitetura do projeto
![bucket_bronze](/assets/arquitetura.jpg)
---

### Visão geral da arquitetura

A arquitetura proposta adota o Databricks como plataforma central de Data Lakehouse, oferecendo um ambiente unificado para ingestão, processamento, governança e consumo de dados.

O uso do Databricks reduz a complexidade operacional ao abstrair a gestão de infraestrutura, permitindo que o time foque na engenharia, qualidade e valor dos dados.

### Justificativa das escolhas

- **Databricks Lakehouse Platform (Free Edition)**  
  Centraliza processamento, armazenamento lógico e ambiente analítico em uma única plataforma, reduzindo overhead operacional e acelerando o desenvolvimento de soluções de dados. A abordagem Lakehouse garante flexibilidade analítica com governança e organização de dados.

- **Apache Spark Gerenciado**
  Permite processamento distribuído e escalável sem a necessidade de administrar clusters manualmente, sendo adequado para ETL/ELT, feature engineering e análises em grandes volumes de dados.

- **Arquitetura Medallion (Bronze, Silver, Gold)** 
  Garante rastreabilidade, qualidade e confiabilidade dos dados. A separação por camadas permite reprocessamentos seguros, auditoria e entrega de dados prontos para consumo analítico e tomada de decisão.

- **Notebooks Databricks**  
  Fornecem um ambiente colaborativo para exploração, validação e desenvolvimento analítico o que acelera o ciclo de análise e experimentação.

---

## Como subir o projeto?
<p> Criar conta no Databricks Free (https://login.databricks.com/signup?) </p>
<p> Criar catalog = hackathon2025: </p>

![catalog](/assets/create-catalog.jpg)

<p> Criar volume e fazer ingestão dentro do catalog hackathon2025: </p>

![catalog](/assets/upload-volumes.jpg)

<p> Criar git folder = Hackathon-2025: </p>

![catalog](/assets/git-folder.jpg)

<p> Criar folder (silver e gold) no workspace Hackathon-2025: </p>

![catalog](/assets/workspace.jpg)