import pickle
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from IPython.display import display
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Avisos.
from warnings import filterwarnings
filterwarnings('ignore')



def normalize_dtypes(df):
    """
    Converte colunas do tipo 'category' para 'object' e unifica tipos num�ricos:
    - Inteiros para int32
    - Floats para float32

    :param df: DataFrame a ser processado
    :return: DataFrame com os tipos de dados padronizados
    """
    df = df.copy()

    # Converter 'category' para 'object' (string)
    cat_cols = df.select_dtypes(include=['category']).columns
    df[cat_cols] = df[cat_cols].astype(str)

    # Converter colunas inteiras para int32
    int_cols = df.select_dtypes(include=['Int16', 'Int32']).columns
    df[int_cols] = df[int_cols].astype('int32')

    # Converter colunas float para float32
    float_cols = df.select_dtypes(include=['float16', 'float32']).columns
    df[float_cols] = df[float_cols].astype('float32')

    return df

def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if (col_type != object) and (str(col_type) != 'category'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                int_types = [
                    (np.int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
                    (np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
                    (np.int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
                    (np.uint16, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
                    (np.int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
                    (np.uint32, np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
                    (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
                    (np.uint64, np.iinfo(np.uint64).min, np.iinfo(np.uint64).max)
                ]
                for dtype, min_val, max_val in int_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break
            elif str(col_type)[:5] == 'float':
                float_types = [
                    (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
                    (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
                    (np.float64, np.finfo(np.float64).min, np.finfo(np.float64).max)
                ]
                for dtype, min_val, max_val in float_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def clf_metric_report(y_score, y_true):
    print('Evaluating the model...')
    roc_auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    logloss = log_loss(y_true, y_score)
    print(f'ROC AUC: {roc_auc}')
    print(f'Brier Score: {brier}')
    print(f'Average Precision: {avg_precision}')
    print(f'Log Loss: {logloss}')

def compute_and_plot_permutation_importance(model, X_test, y_test, metric='average_precision', n_repeats=5):
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring=metric)
    features = X_test.columns.to_list()

    # Sort features by importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': result.importances_mean})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Plot top 20 most important features using seaborn
    plt.figure(figsize=(10, 12))
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()
    return feature_importance

def plot_calibration_curve(y_score, y_true, title='Calibration Curve'):
    prob_true, prob_pred = calibration_curve(y_score, y_true, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.show()

def plot_pr_calib_curve(y_score, y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')

    plt.tight_layout()
    plt.show()

def plot_dis_probs(y_score, y_true):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_score[y_true == 1], bins=50, color='red', label='Churn', kde=True, stat='density')
    sns.histplot(y_score[y_true == 0], bins=50, color='blue', label='Non-Churn', kde=True, stat='density')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Churn vs Non-Churn')
    plt.legend()
    plt.show()


import numpy as np



def reduce_mem_usage_spark(df, verbose=False):
    # Iterando sobre as colunas do DataFrame
    for col in df.columns:
        # Pegando o tipo da coluna
        col_type = dict(df.dtypes)[col]
        
        # Calculando os valores mínimo e máximo da coluna
        c_min, c_max = df.select(F.min(col), F.max(col)).first()

        if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
            if 'int' in col_type:
                int_types = [
                    (IntegerType(), -2147483648, 2147483647),  # int32
                    (LongType(), -9223372036854775808, 9223372036854775807),  # int64
                ]
                for dtype, min_val, max_val in int_types:
                    if c_min > min_val and c_max < max_val:
                        df = df.withColumn(col, df[col].cast(dtype))
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

            elif 'float' in col_type:
                float_types = [
                    (FloatType(), -3.4028235e+38, 3.4028235e+38),  # float32
                    (DoubleType(), -1.7976931348623157e+308, 1.7976931348623157e+308),  # float64
                ]
                for dtype, min_val, max_val in float_types:
                    if c_min > min_val and c_max < max_val:
                        df = df.withColumn(col, df[col].cast(dtype))
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

    return df

# Função para verificar os metadados da tabela.
def generate_metadata(df, ids=None, targets=None, orderby='PC_NULOS'):
    """
    Esta função retorna uma tabela com informações descritivas sobre um DataFrame.

    Parâmetros:
    - df: DataFrame que você quer descrever.
    - ids: Lista de colunas que são identificadores.
    - targets: Lista de colunas que são variáveis alvo.
    - orderby: Coluna pela qual ordenar os resultados.

    Retorna:
    Um DataFrame com informações sobre o df original.
    """

    if ids is None:
        summary = pd.DataFrame({
            'FEATURE': df.columns,
            'USO_FEATURE': ['Target' if col in targets else 'Explicativa' for col in df.columns],
            'QT_NULOS': df.isnull().sum(),
            'PC_NULOS': round((df.isnull().sum() / len(df)) * 100, 2),
            'CARDINALIDADE': df.nunique(),
            'TIPO_FEATURE': df.dtypes
        })
    else:
        summary = pd.DataFrame({
            'FEATURE': df.columns,
            'USO_FEATURE': ['ID' if col in ids else 'Target' if col in targets else 'Explicativa' for col in df.columns],
            'QT_NULOS': df.isnull().sum(),
            'PC_NULOS': round((df.isnull().sum() / len(df)) * 100, 2),
            'CARDINALIDADE': df.nunique(),
            'TIPO_FEATURE': df.dtypes
        })

    summary_sorted = summary.sort_values(by=orderby, ascending=False)
    summary_sorted = summary_sorted.reset_index(drop=True)

    return summary_sorted



# =================================================================================================================================================== #



def custom_fillna(df):
    '''
    Esta função preenche os valores nulos do DataFrame com a média das colunas numéricas e com 'VERIFICAR' para as colunas categóricas.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido, um dicionário contendo as médias das colunas numéricas e outro dicionário com os valores usados para preencher as colunas categóricas.
    '''

    # Preenchimento para colunas numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col], inplace=True)

    # Preenchimento para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col], inplace=True)

    return df, means, modes


# Função para preenchimento dos valores nulos em produção.
def custom_fillna_prod(df, means):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.

    Retorna:
    O DataFrame preenchido.
    '''

    for col, mean_value in means.items():
      df[col].fillna(mean_value, inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('VERIFICAR')

    return df

# Função para o cálculo do WoE e IV.
def calculate_woe_iv(df, feature, target, bins=10):
    """
    Calcula WOE (Weight of Evidence) e IV (Information Value) para uma variável.
    Se a variável for numérica, aplica binning automático.
    
    Parâmetros:
    - df: DataFrame com os dados
    - feature: nome da variável
    - target: nome da variável target (binária: 0 e 1)
    - bins: número de bins para variáveis numéricas (default=5)
    
    Retorna:
    - iv (float)
    """
    
    # Trata valores ausentes
    df_temp = df[[feature, target]].copy()
    df_temp = df_temp.dropna(subset=[feature, target])

    # Se for numérica, aplica binning
    if pd.api.types.is_numeric_dtype(df_temp[feature]):
        try:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')
        except:
            # fallback para cut se qcut falhar
            df_temp['bin'] = pd.cut(df_temp[feature], bins=bins)
    else:
        df_temp['bin'] = df_temp[feature].astype(str)

    # Contagem de bons e maus por bin
    lst = []
    for val in df_temp['bin'].unique():
        total = df_temp[df_temp['bin'] == val].shape[0]
        good = df_temp[(df_temp['bin'] == val) & (df_temp[target] == 1)].shape[0]
        bad = df_temp[(df_temp['bin'] == val) & (df_temp[target] == 0)].shape[0]
        lst.append({
            'Value': val,
            'All': total,
            'Good': good,
            'Bad': bad
        })

    dset = pd.DataFrame(lst)

    # Calcula distribuições
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

    # Evita divisão por zero
    dset['Distr_Good'] = dset['Distr_Good'].replace(0, 1e-6)
    dset['Distr_Bad'] = dset['Distr_Bad'].replace(0, 1e-6)

    # Calcula WoE e IV
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

    return dset['IV'].sum()





def iv_table(df, target):
    """
    Retorna uma tabela com IV para todas as variáveis em relação ao target.
    """
    iv_list = []
    for col in df.columns:
        if col == target:
            continue
        iv = calculate_woe_iv(df, col, target)
        if iv < 0.02:
            predictiveness = 'Inútil para a predição'
        elif iv < 0.1:
            predictiveness = 'Preditor Fraco'
        elif iv < 0.3:
            predictiveness = 'Preditor Moderado'
        else:
            predictiveness = 'Preditor Forte'
        iv_list.append({
            'Variável': col,
            'IV': iv,
            'Preditividade': predictiveness
        })

    return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)




# =================================================================================================================================================== #



# Função para calcular o KS.
def calcular_ks_statistic(y_true, y_score):
    '''
    Calcula o KS (Kolmogorov-Smirnov) para avaliação de um modelo de classificação.

    Parâmetros:
    - y_true: Valores verdadeiros.
    - y_score: Escores previstos.

    Retorna:
    - Valor do KS.
    '''

    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values(by='score', ascending=False)
    total_events = df.target.sum()
    total_non_events = len(df) - total_events
    df['cum_events'] = df.target.cumsum()
    df['cum_non_events'] = (df.target == 0).cumsum()
    df['cum_events_percent'] = df.cum_events / total_events
    df['cum_non_events_percent'] = df.cum_non_events / total_non_events
    ks_statistic = np.abs(df.cum_events_percent - df.cum_non_events_percent).max()
    return ks_statistic


# Função para calcular o KS.
def gini_normalizado(y_true, y_pred):

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


# Função para calcular as métricas e plotar.
def avaliar_modelo(X_train, y_train, X_test, y_test, modelo, nm_modelo):
    '''
    Avalia um modelo de classificação e plota várias métricas de desempenho.

    Parâmetros:
    - X_train: Features do conjunto de treino.
    - y_train: Variável alvo do conjunto de treino.
    - X_test: Features do conjunto de teste.
    - y_test: Variável alvo do conjunto de teste.
    - modelo: Modelo treinado.
    - nm_modelo: Nome do modelo.

    Retorna:
    Uma série de gráficos com as principais métricas de desempenho para treino e teste.
    '''

    feature_names = list(X_train.columns)
    # Criação da figura e dos eixos.
    fig, axs = plt.subplots(5, 2, figsize=(15, 30))     # Ajustado para incluir novos gráficos.
    plt.tight_layout(pad=6.0)

    # Cor azul claro.
    cor = 'skyblue'

    # Taxa de Evento e Não Evento.
    event_rate = np.mean(y_train)
    non_event_rate = 1 - event_rate
    axs[0, 0].bar(['Evento', 'Não Evento'], [event_rate, non_event_rate], color=[cor, 'lightcoral'])
    axs[0, 0].set_title('Taxa de Evento e Não Evento')
    axs[0, 0].set_ylabel('Proporção')

    # Importância dos Atributos.
    importancias = None
    if hasattr(modelo, 'coef_'):      # hasattr = Tem atributo? Se tem coeficiênte ou não, se não tiver ele calcula a feature importance, sem tem coeficiênte, tem beta, ele calcula o peso do beta.
        importancias = np.abs(modelo.coef_[0])
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_

    if importancias is not None:
        importancias_df = pd.DataFrame({'feature': feature_names, 'importance': importancias})
        importancias_df = importancias_df.sort_values(by='importance', ascending=True)

        axs[0, 1].barh(importancias_df['feature'], importancias_df['importance'], color=cor)
        axs[0, 1].set_title('Importância das Variáveis - ' + nm_modelo)
        axs[0, 1].set_xlabel('Importância')

    else:
        axs[0, 1].axis('off')     # Desativa o subplot se não houver importâncias para mostrar.

    # Confusion Matrix - Treino.
    y_pred_train = modelo.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    axs[1, 0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix - Treino - ' + nm_modelo)
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels(['0', '1'])
    axs[1, 0].set_yticklabels(['0', '1'])
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        axs[1, 0].text(j, i, format(cm_train[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    # Confusion Matrix - Teste.
    y_pred_test = modelo.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    axs[1, 1].imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 1].set_title('Confusion Matrix - Teste - ' + nm_modelo)
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(['0', '1'])
    axs[1, 1].set_yticklabels(['0', '1'])
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        axs[1, 1].text(j, i, format(cm_test[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    # ROC Curve - Treino e Teste.
    y_score_train = modelo.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    axs[2, 0].plot(fpr_train, tpr_train, color=cor, label='Treino')

    y_score_test = modelo.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    axs[2, 0].plot(fpr_test, tpr_test, color='darkorange', label='Teste')

    axs[2, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[2, 0].set_title('ROC Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 0].set_xlabel('False Positive Rate')
    axs[2, 0].set_ylabel('True Positive Rate')
    axs[2, 0].legend(loc="lower right")

    # Precision-Recall Curve - Treino e Teste.
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    axs[2, 1].plot(recall_train, precision_train, color=cor, label='Treino')

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_score_test)
    axs[2, 1].plot(recall_test, precision_test, color='darkorange', label='Teste')

    axs[2, 1].set_title('Precision-Recall Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 1].set_xlabel('Recall')
    axs[2, 1].set_ylabel('Precision')
    axs[2, 1].legend(loc="upper right")

    # Gini - Treino e Teste.
    auc_train = roc_auc_score(y_train, y_score_train)
    gini_train = 2 * auc_train - 1
    auc_test = roc_auc_score(y_test, y_score_test)
    gini_test = 2 * auc_test - 1
    axs[3, 0].bar(['Treino', 'Teste'], [gini_train, gini_test], color=[cor, 'darkorange'])
    axs[3, 0].set_title('Gini - ' + nm_modelo)
    axs[3, 0].set_ylim(0, 1)
    axs[3, 0].text('Treino', gini_train + 0.01, f'{gini_train:.2f}', ha='center', va='bottom')
    axs[3, 0].text('Teste', gini_test + 0.01, f'{gini_test:.2f}', ha='center', va='bottom')

    # KS - Treino e Teste.
    ks_train = calcular_ks_statistic(y_train, y_score_train)
    ks_test = calcular_ks_statistic(y_test, y_score_test)
    axs[3, 1].bar(['Treino', 'Teste'], [ks_train, ks_test], color=[cor, 'darkorange'])
    axs[3, 1].set_title('KS - ' + nm_modelo)
    axs[3, 1].set_ylim(0, 1)
    axs[3, 1].text('Treino', ks_train + 0.01, f'{ks_train:.2f}', ha='center', va='bottom')
    axs[3, 1].text('Teste', ks_test + 0.01, f'{ks_test:.2f}', ha='center', va='bottom')

    # Decile Analysis - Teste.
    scores = modelo.predict_proba(X_test)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores.shape)     # Adiciona um pequeno ruído.
    scores += noise
    deciles = pd.qcut(scores, q=10, duplicates='drop')
    decile_analysis = y_test.groupby(deciles).mean()
    axs[4, 1].bar(range(1, len(decile_analysis) + 1), decile_analysis, color='darkorange')
    axs[4, 1].set_title('Ordenação do Score - Teste - ' + nm_modelo)
    axs[4, 1].set_xlabel('Faixas de Score')
    axs[4, 1].set_ylabel('Taxa de Evento')

    # Decile Analysis - Treino.
    scores_train = modelo.predict_proba(X_train)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores_train.shape)     # Adiciona um pequeno ruído.
    scores_train += noise
    deciles_train = pd.qcut(scores_train, q=10, duplicates='drop')
    decile_analysis_train = y_train.groupby(deciles_train).mean()
    axs[4, 0].bar(range(1, len(decile_analysis_train) + 1), decile_analysis_train, color=cor)
    axs[4, 0].set_title('Ordenação do Score - Treino - ' + nm_modelo)
    axs[4, 0].set_xlabel('Faixas de Score')
    axs[4, 0].set_ylabel('Taxa de Evento')

    # Mostrar os gráficos.
    plt.show()



# =================================================================================================================================================== #



# Função para criar um DataFrame com as métricas de todos os modelos treinados.
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''
    Avalia múltiplos modelos de classificação e retorna um DataFrame com as métricas de desempenho de cada modelo, destacando as métricas mais altas.

    Parâmetros:
    - X_train: Features do conjunto de treino.
    - y_train: Variável alvo do conjunto de treino.
    - X_test: Features do conjunto de teste.
    - y_test: Variável alvo do conjunto de teste.
    - models: Dicionário contendo os modelos treinados.

    Retorna:
    DataFrame contendo as métricas de desempenho de todos os modelos.
    '''

    metrics = []
    for name, model in models.items():
        # Iniciando o cronômetro.
        import time
        start_time = time.time()

        # Prever os rótulos para os conjuntos de treino e teste.
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Calcular as métricas.
        accuracy = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])     # Supondo que é um problema de classificação binária.
        gini = 2*auc - 1
        ks = calcular_ks_statistic(y_test, model.predict_proba(X_test)[:, 1])

        # Calculando o tempo de treinamento.
        end_time = time.time()
        training_time = end_time - start_time

        # Adicionar ao array de métricas.
        metrics.append({
            'Model': name,
            'AUC-ROC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Gini': gini,
            'KS': ks,
            'Training_Time(s)': training_time
        })

    # Convertendo o array de métricas em um DataFrame.
    metrics_df = pd.DataFrame(metrics)

    # Ordenando o DataFrame pela metrica AUC-ROC.
    metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

    # Função para destacar o maior valor em azul claro.
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightblue' if v else '' for v in is_max]

    # Destacando o maior valor de cada métrica
    metrics_df_max = metrics_df_sorted.style.apply(highlight_max, subset=metrics_df.columns[1:-1])

    return metrics_df_max

def plot_tx_event_volume_safra(dataframe, target, safra, ymax_volume=None, ymax_taxa_evento=None):


  # Converte a coluna 'var' para string se seu tipo não for string
  if dataframe[safra].dtype != 'O':  
      dataframe[safra] = dataframe[safra].astype(str)

  # Calcule a média do TARGET_60_6 e o volume por AAAAMM
  resultado = dataframe.groupby(safra).agg({target: 'mean', safra: 'count'}).rename(columns={safra: 'Volume'}).reset_index()
  resultado.columns = ['Safra', 'Taxa_de_Evento', 'Volume']

  # Gráfico com barras para o volume e linha para a taxa de evento por safra (AAAAMM)
  fig, ax1 = plt.subplots(figsize=(12, 6))

  color = 'lightblue'
  ax1.bar(resultado['Safra'], resultado['Volume'], color=color, label='Volume')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')
  if ymax_volume:
      ax1.set_ylim(0, ymax_volume)

  ax2 = ax1.twinx()
  color = 'hotpink'
  ax2.plot(resultado['Safra'], resultado['Taxa_de_Evento'] * 100, marker='o', linestyle='-', color=color, label='Taxa de Evento (%)')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')
  if ymax_taxa_evento:
      ax2.set_ylim(0, ymax_taxa_evento)

  for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
      label.set_fontsize(7)
      label.set_color('black')

  plt.title('Volume e Taxa de Evento por Safra')
  plt.legend(loc='upper left')
  plt.show()

  return resultado

def analyze_variable(dataframe, variable, target):


  # Se a variável for numérica, arredonda para 4 casas decimais
  if pd.api.types.is_numeric_dtype(dataframe[variable]):
      dataframe[variable] = dataframe[variable].round(4)
      dataframe[variable] = dataframe[variable].astype(str)

  # Calcula a taxa de evento e o volume para cada categoria da variável
  result = dataframe.groupby(variable).agg({target: 'mean', variable: 'count'}).rename(columns={variable: 'Volume'}).reset_index()
  result.columns = [variable, 'Taxa_de_Evento', 'Volume']

  # Ordena o resultado pela Taxa de Evento em ordem decrescente
  result = result.sort_values(by='Taxa_de_Evento', ascending=False)

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(12, 6))

  # Eixo Y esquerdo: Volume
  bars = ax1.bar(result[variable], result['Volume'], color='lightblue', label='Volume (Barras)')
  ax1.set_xlabel(variable)
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento
  ax2 = ax1.twinx()
  lines = ax2.plot(result[variable], result['Taxa_de_Evento'] * 100, marker='o', linestyle='-', color='hotpink', label='Taxa de Evento (Linha)')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  # Combina as legendas de ambos os eixos, filtrando rótulos que começam com '_'
  plots = [item for item in bars + tuple(lines) if not item.get_label().startswith('_')]
  labels = [plot.get_label() for plot in plots]
  plt.legend(plots, labels, loc='upper left')

  plt.title(f'Volume e Taxa de Evento por {variable}')
  plt.xticks(rotation=45)  # Adicionado para melhor visualização dos labels no eixo X
  plt.tight_layout()
  plt.show()

  return result


def plot_by_safra(dataframe, target, explicativa, safra):


  df_copy = dataframe.copy()
  # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
      df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
  result = df_copy.groupby([safra, explicativa]).agg({target: 'mean', explicativa: 'count'}).rename(columns={explicativa: 'Volume'}).reset_index()

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(15, 10))

  # Eixo Y esquerdo: Volume total por safra
  volume_by_safra = result.groupby(safra).agg({'Volume': 'sum'}).reset_index()
  bars = ax1.bar(volume_by_safra[safra], volume_by_safra['Volume'], color='lightblue', label='Volume Total (Barras)')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento por categoria
  ax2 = ax1.twinx()
  for category in result[explicativa].unique():
      subset = result[result[explicativa] == category]
      ax2.plot(subset[safra], subset[target] * 100, marker='o', linestyle='-', label=f'Taxa de Evento ({category})')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  plt.title(f'Volume Total e Taxa de Evento por {explicativa} ao longo das Safras')
  plt.legend(loc='upper left')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()



def group_and_plot_by_safra(dataframe, target, explicativa, safra, domain_mapping):


  df_copy = dataframe.copy()
  # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
      df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Cria uma coluna com os valores originais para mostrar a transformação posteriormente
  df_copy['original_' + explicativa] = df_copy[explicativa]

  # Aplica o mapeamento para os novos domínios
  df_copy[explicativa] = df_copy[explicativa].map(domain_mapping).fillna(df_copy[explicativa])

  # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
  result = df_copy.groupby([safra, explicativa]).agg({target: 'mean', explicativa: 'count'}).rename(columns={explicativa: 'Volume'}).reset_index()

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(12, 6))

  # Eixo Y esquerdo: Volume total por safra
  volume_by_safra = result.groupby(safra).agg({'Volume': 'sum'}).reset_index()
  bars = ax1.bar(volume_by_safra[safra], volume_by_safra['Volume'], color='lightblue', label='Volume Total (Barras)')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento por categoria
  ax2 = ax1.twinx()
  for category in result[explicativa].unique():
      subset = result[result[explicativa] == category]
      ax2.plot(subset[safra], subset[target] * 100, marker='o', linestyle='-', label=f'Taxa de Evento ({category})')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  plt.title(f'Volume Total e Taxa de Evento por {explicativa} ao longo das Safras')
  plt.legend(loc='upper left')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  # Cria um DataFrame de transformação
  transformation_df = df_copy[[explicativa, 'original_' + explicativa]].drop_duplicates().sort_values(by='original_' + explicativa)
  transformation_df.rename(columns={explicativa:'TFB_'+explicativa,'original_' + explicativa:explicativa},inplace=True)

  return transformation_df


def apply_grouping(data, transformation_df, explicativa):

  df_copy = data.copy()

  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
    df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Une o DataFrame de transformação com os novos dados para aplicar a transformação
  df_copy = df_copy.merge(transformation_df, left_on=explicativa, right_on=explicativa, how='left')

  # Aplica a transformação
  colname_transformed = 'TFB_' + explicativa
  df_copy[explicativa] = df_copy[colname_transformed].fillna(df_copy[explicativa])

  # Remove a coluna original
  df_copy.drop(columns=[explicativa], inplace=True)

  return df_copy

def categorize_with_decision_tree(dataframe, n_categories, target, numeric_var):

  
  # Preparar os dados
  X = dataframe[[numeric_var]]
  y = dataframe[target]

  # Treinar uma árvore de decisão com profundidade máxima igual ao número de categorias desejadas
  tree = DecisionTreeClassifier(max_leaf_nodes=n_categories)
  tree.fit(X, y)

  # Predizer a categoria (folha) para cada entrada no DataFrame
  leaf_ids = tree.apply(X)

  # Criar um DataFrame temporário com as categorias (folhas), a variável numérica e o target
  temp_df = pd.DataFrame({numeric_var: dataframe[numeric_var], 'Leaf': leaf_ids, target: y})

  result = temp_df.groupby('Leaf').agg({target: 'mean', numeric_var: ['count', 'min', 'max']}).reset_index()
  result.columns = ['Leaf', 'Taxa_de_Evento', 'Volume', 'Lower_Bound', 'Upper_Bound']

  # Ajuste para garantir que os limites superior e inferior de bins adjacentes não se sobreponham
  result = result.sort_values(by='Lower_Bound')
  for i in range(1, len(result)):
      result.iloc[i, 3] = max(result.iloc[i, 3], result.iloc[i-1, 4])

  # Definir o limite inferior do primeiro bin como -inf e o limite superior do último bin como inf
  result.iloc[0, 3] = -np.inf
  result.iloc[-1, 4] = np.inf

  return result


def apply_tree_bins(data, transformation_df, numeric_var):

  import numpy as np
  df_copy = data.copy()

  # Obtenha os limites superiores e ordene-os
  upper_bounds = transformation_df['Upper_Bound'].sort_values().values

  # Use numpy.digitize para determinar a qual bin cada valor pertence
  df_copy[f"TFT_{numeric_var}"] = np.digitize(df_copy[numeric_var].values, upper_bounds)
  df_copy.drop(axis=1,columns=[numeric_var],inplace=True)

  return df_copy





def calculate_ks(y_true, y_score):
  from sklearn.metrics import roc_auc_score, roc_curve
  """Calculate KS statistic."""
  fpr, tpr, _ = roc_curve(y_true, y_score)
  return 100*max(tpr - fpr)

def calculate_gini(y_true, y_score):
  from sklearn.metrics import roc_auc_score, roc_curve
  """Calculate Gini coefficient."""
  return (2 * roc_auc_score(y_true, y_score) - 1)*100


def plot_ks_gini_by_datref(df, target_col, score_col, datref_col,titulo='KS e GIni por Safra'):

  df[datref_col] = df[datref_col].astype(str)
  unique_dates = sorted(df[datref_col].unique())
  ks_values = []
  gini_values = []

  for date in unique_dates:
      subset = df[df[datref_col] == date]
      y_true = subset[target_col]
      y_score = subset[score_col]

      ks_values.append(calculate_ks(y_true, y_score))
      gini_values.append(calculate_gini(y_true, y_score))

  plt.figure(figsize=(12, 6))
  plt.plot(unique_dates, ks_values, label='KS', marker='o')
  plt.plot(unique_dates, gini_values, label='Gini', marker='o')
  plt.xlabel(datref_col)
  plt.ylabel('Value')
  plt.ylim(0, 100)
  plt.title(titulo)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def apply_fillna(df):
    '''
    Esta função preenche os valores nulos do DataFrame com a média das colunas numéricas 
    e com a moda para as colunas categóricas.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e dois dicionários:
    - means: contendo as médias das colunas numéricas.
    - modes: contendo as modas das colunas categóricas.
    '''

    # Preenchimento para colunas numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col])

    # Preenchimento para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col])

    return df, means, modes


def apply_fillna_prod(df, means, modes):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.
    - modes: Dicionário contendo as modas das colunas categóricas.

    Retorna:
    O DataFrame preenchido.
    '''

    # Preenchimento para colunas numéricas
    for col, mean_value in means.items():
        df[col].fillna(mean_value, inplace=True)

    # Preenchimento para colunas categóricas
    for col, mode_value in modes.items():
        if col in df.columns:
            df[col].fillna(mode_value, inplace=True)

    return df

def iv_category(df, target, variable, count):

    pivot_table = pd.pivot_table(df, columns=target, index=variable, values=count, aggfunc="count")
    pivot_table['%Sim'] = pivot_table['Sim'].apply(lambda x: round(x/pivot_table['Sim'].sum()*100,2))
    pivot_table['%Nao'] = pivot_table['Não'].apply(lambda x: round(x/pivot_table['Não'].sum()*100,2))
    pivot_table['%FreqLinha'] = (pivot_table['Sim'] / (pivot_table['Não'] + pivot_table['Sim']) * 100).round(2)
    pivot_table['WoE-ODDS'] = (pivot_table['%Sim'] / pivot_table['%Nao'])
    pivot_table['IV'] = ((pivot_table['%Sim']/100) - (pivot_table['%Nao']/100)) * np.log(pivot_table['WoE']).replace([np.inf, -np.inf], 0)
    
    return pivot_table


def iv_bin(df, target, variable, bins, count):

    bin_variable = pd.cut(x=df[variable], bins=bins)
    
    pivot_table = pd.pivot_table(df, columns=target, index=bin_variable, values=count, aggfunc="count")
    pivot_table['%Sim'] = pivot_table['Sim'].apply(lambda x: round(x/pivot_table['Sim'].sum()*100,2))
    pivot_table['%Nao'] = pivot_table['Não'].apply(lambda x: round(x/pivot_table['Não'].sum()*100,2))
    pivot_table['%FreqLinha'] = (pivot_table['Sim'] / (pivot_table['Não'] + pivot_table['Sim']) * 100).round(2)
    pivot_table['WoE'] = (pivot_table['%Sim'] / pivot_table['%Nao'])
    pivot_table['IV'] = ((pivot_table['%Sim']/100) - (pivot_table['%Nao']/100)) * np.log(pivot_table['WoE']).replace([np.inf, -np.inf], 'Não')

    return pivot_table

def metrics_calculate(nm_modelo, model, X_train, y_train, X_test, y_test):
    # Fazendo predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculando as métricas para o conjunto de treino
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    auc_roc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    # Calculando o Índice Gini e Estatística KS para o conjunto de treino
    probabilities_train = model.predict_proba(X_train)[:, 1]
    df_train = pd.DataFrame({'true_labels': y_train, 'predicted_probs': probabilities_train})
    df_train = df_train.sort_values(by='predicted_probs', ascending=False)
    df_train['cumulative_true'] = df_train['true_labels'].cumsum() / df_train['true_labels'].sum()
    df_train['cumulative_false'] = (1 - df_train['true_labels']).cumsum() / (1 - df_train['true_labels']).sum()
    ks_statistic_train = max(abs(df_train['cumulative_true'] - df_train['cumulative_false']))
    gini_index_train = 2 * auc_roc_train - 1

    # Calculando as métricas para o conjunto de teste
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Calculando o Índice Gini e Estatística KS para o conjunto de teste
    probabilities_test = model.predict_proba(X_test)[:, 1]
    df_test = pd.DataFrame({'true_labels': y_test, 'predicted_probs': probabilities_test})
    df_test = df_test.sort_values(by='predicted_probs', ascending=False)
    df_test['cumulative_true'] = df_test['true_labels'].cumsum() / df_test['true_labels'].sum()
    df_test['cumulative_false'] = (1 - df_test['true_labels']).cumsum() / (1 - df_test['true_labels']).sum()
    ks_statistic_test = max(abs(df_test['cumulative_true'] - df_test['cumulative_false']))
    gini_index_test = 2 * auc_roc_test - 1

    # Criando o DataFrame com as métricas calculadas
    metrics_df = pd.DataFrame({
        'Algoritmo': [nm_modelo, nm_modelo],
        'Conjunto': ['Treino', 'Teste'],
        'Acuracia': [accuracy_train, accuracy_test],
        'Precisao': [precision_train, precision_test],
        'Recall': [recall_train, recall_test],
        'AUC_ROC': [auc_roc_train, auc_roc_test],
        'GINI': [gini_index_train, gini_index_test],
        'KS': [ks_statistic_train, ks_statistic_test]
    })
    return metrics_df

def fillna_numeric(df):
    '''
    Preenche os valores nulos das colunas numéricas do DataFrame com a média de cada coluna.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e um dicionário contendo as médias das colunas numéricas.
    '''
    df.replace(-1, np.nan, inplace=True)
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'float16', 'int8', 'uint8', 'uint16', 'uint32', 'uint64', 'int16']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col], inplace=True)
    return df, means

def fillna_categorical(df):
    '''
    Preenche os valores nulos das colunas categóricas do DataFrame com o modo de cada coluna ou com 'VERIFICAR' se não houver um modo.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e um dicionário contendo os valores usados para preencher as colunas categóricas.
    '''
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col], inplace=True)
    return df, modes

# Função para preenchimento dos valores nulos em produção.
def fillna_num_prod(df, means):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.

    Retorna:
    O DataFrame preenchido.
    '''
    df.replace(-1, np.nan, inplace=True)
    for col, mean_value in means.items():
      df[col].fillna(mean_value, inplace=True)

    return df

# Função para preenchimento dos valores nulos em produção.
def fillna_catg_prod(df, modes):
    '''
    Esta função preenche os valores nulos das colunas categóricas do DataFrame em produção
    com os valores fornecidos em um dicionário de modos.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - modes: Dicionário contendo os valores (modos) para preencher as colunas categóricas.

    Retorna:
    O DataFrame preenchido.
    '''
    for col, mode_value in modes.items():
        if col in df.columns:  # Verifica se a coluna existe no DataFrame
            df[col].fillna(mode_value, inplace=True)

    return df


def calculate_metrics_models_classifier(nm_modelo, model, X_train, y_train, X_test, y_test):
    # Fazendo predi��es
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculando as m�tricas para o conjunto de treino
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    auc_roc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    # Calculando o �ndice Gini e Estat�stica KS para o conjunto de treino
    probabilities_train = model.predict_proba(X_train)[:, 1]
    df_train = pd.DataFrame({'true_labels': y_train, 'predicted_probs': probabilities_train})
    df_train = df_train.sort_values(by='predicted_probs', ascending=False)
    df_train['cumulative_true'] = df_train['true_labels'].cumsum() / df_train['true_labels'].sum()
    df_train['cumulative_false'] = (1 - df_train['true_labels']).cumsum() / (1 - df_train['true_labels']).sum()
    ks_statistic_train = max(abs(df_train['cumulative_true'] - df_train['cumulative_false']))
    gini_index_train = 2 * auc_roc_train - 1

    # Calculando as m�tricas para o conjunto de teste
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Calculando o �ndice Gini e Estat�stica KS para o conjunto de teste
    probabilities_test = model.predict_proba(X_test)[:, 1]
    df_test = pd.DataFrame({'true_labels': y_test, 'predicted_probs': probabilities_test})
    df_test = df_test.sort_values(by='predicted_probs', ascending=False)
    df_test['cumulative_true'] = df_test['true_labels'].cumsum() / df_test['true_labels'].sum()
    df_test['cumulative_false'] = (1 - df_test['true_labels']).cumsum() / (1 - df_test['true_labels']).sum()
    ks_statistic_test = max(abs(df_test['cumulative_true'] - df_test['cumulative_false']))
    gini_index_test = 2 * auc_roc_test - 1

    # Criando o DataFrame com as m�tricas calculadas
    metrics_df = pd.DataFrame({
        'Algoritmo': [nm_modelo, nm_modelo],
        'Conjunto': ['Treino', 'Teste'],
        #'Acuracia': [accuracy_train, accuracy_test],
        #'Precisao': [precision_train, precision_test],
        #'Recall': [recall_train, recall_test],
        'AUC_ROC': [auc_roc_train, auc_roc_test],
        'GINI': [gini_index_train, gini_index_test],
        'KS': [ks_statistic_train, ks_statistic_test]
    })
    return metrics_df




def calculate_metrics_models_regression(nm_modelo, model, X_train, y_train, X_test, y_test):
    # Fazendo predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculando métricas para o conjunto de treino
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

    # Calculando métricas para o conjunto de teste
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    # Criando o DataFrame com as métricas calculadas
    metrics_df = pd.DataFrame({
        'Algoritmo': [nm_modelo, nm_modelo],
        'Conjunto': ['Treino', 'Teste'],
        'R2': [r2_train, r2_test],
        'MAE': [mae_train, mae_test],
        'MSE': [mse_train, mse_test],
        'RMSE': [rmse_train, rmse_test],
        'MAPE (%)': [mape_train, mape_test]
    })

    return metrics_df

class analise_iv:
        
    # função private
    def __get_tab_bivariada(self, var_escolhida):
     
        # Cria a contagem de Target_1 e Target_0
        df_aux = self.df.copy() 
        df_aux['target2'] = self.df[self.target]
        df2 = df_aux.pivot_table(values='target2',
                                 index=var_escolhida,
                                 columns=self.target,
                                 aggfunc='count')
        
        df2 = df2.rename(columns={0:'#Target_0',
                                  1:'#Target_1'})
        df2.fillna(0, inplace=True)

        # Cria as demais colunas da tabela bivariada
        df2['Total'] = (df2['#Target_0'] + df2['#Target_1'])
        df2['%Freq'] = (df2['Total'] / (df2['Total'].sum()) * 100).round(decimals=2)
        df2['%Target_1'] = (df2['#Target_1'] / (df2['#Target_1'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = (df2['#Target_0'] / (df2['#Target_0'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = df2['%Target_0'].apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['%Taxa_de_Target_1'] = (df2['#Target_1'] / df2['Total'] * 100).round(decimals=2)
        df2['Odds'] = (df2['%Target_1'] / df2['%Target_0']).round(decimals=2)
        df2['Odds'] = df2.Odds.apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['LN(Odds)'] = np.log(df2['Odds']).round(decimals=2)
        df2['IV'] = (((df2['%Target_1'] / 100 - df2['%Target_0'] / 100) * df2['LN(Odds)'])).round(decimals=2)
        df2['IV'] = np.where(df2['Odds'] == 0.01, 0 , df2['IV']) 

        df2 = df2.reset_index()
        df2['Variavel'] = var_escolhida
        df2 = df2.rename(columns={var_escolhida: 'Var_Range'})
        df2 = df2[['Variavel','Var_Range', '#Target_1','#Target_0', 'Total', '%Freq', '%Target_1', '%Target_0',
       '%Taxa_de_Target_1', 'Odds', 'LN(Odds)', 'IV']]
        
        # Guarda uma cópia da tabela no histórico
        self.df_tabs_iv = pd.concat([self.df_tabs_iv, df2], axis = 0)
        
        return df2
        
    def get_bivariada(self, var_escolhida='all_vars'):
        
        if var_escolhida == 'all_vars':
                       
            #vars = self.df.drop(self.target,axis = 1).columns
            vars = self.get_lista_iv().index
            for var in vars:
                tabela = self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]
                print('==> "{}" tem IV de {}'.format(var,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]['IV'].sum().round(decimals=2)))
                # printa a tabela no Jupyter
                display(tabela)
            
            return
        
        else:
            print('==> "{}" tem IV de {}'.format(var_escolhida,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]['IV'].sum().round(decimals=2)))
            return self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]
                   
            
    def get_lista_iv(self):
        
    
        # agrupa a lista de IV's em ordem descrescente
        lista = (self.df_tabs_iv.groupby('Variavel').agg({'IV':'sum'})).sort_values(by=['IV'],ascending=False)
            
        return lista
    
    

    def __init__(self, df, target, nbins=10):

        self.df = df.copy()
        self.target = target

        #lista de variaveis numericas
        df_num = self.df.loc[:,((self.df.dtypes == 'int32') | 
                                (self.df.dtypes == 'int64') | 
                                (self.df.dtypes == 'float64')
                               )
                            ]

        vars = df_num.drop(target,axis = 1).columns

        for var in vars:
            nome_var = 'fx_' + var 
            df_num[nome_var] = pd.qcut(df_num[var], 
                                       q=nbins, 
                                       precision=2,
                                       duplicates='drop')
            df_num = df_num.drop(var, axis = 1)
            df_num = df_num.rename(columns={nome_var: var})

        #lista de variaveis qualitativas
        df_str = self.df.loc[:,((self.df.dtypes == 'object') | 
                                (self.df.dtypes == 'category') |
                                (self.df.dtypes == 'bool'))]


        self.df = pd.concat([df_num,df_str],axis = 1)


         # inicializa tab historica
        self.df_tabs_iv = pd.DataFrame()

        vars = self.df.drop(self.target,axis = 1).columns
        for var in vars:
            self.__get_tab_bivariada(var);

        # remove tabs de iv duplicadas
        self.df_tabs_iv = self.df_tabs_iv.drop_duplicates(subset=['Variavel','Var_Range'], keep='last')



# ------------------------------
# Univariada - Categóricas
# ------------------------------
def univariate_categorical(data, cat_vars):
    n = len(cat_vars)
    n_rows = math.ceil(n / 3)

    for i in range(0, n, 3):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for j in range(3):
            if i + j < n:
                col = cat_vars[i + j]
                sns.countplot(data=data, x=col, order=data[col].value_counts().index, palette='viridis', ax=axs[j])
                axs[j].set_title(f'{col}')
                axs[j].tick_params(axis='x', rotation=45)
            else:
                axs[j].axis('off')
        plt.tight_layout()
        plt.show()

# ------------------------------
# Univariada - Numéricas
# ------------------------------
def univariate_numerical(data, num_vars):
    n = len(num_vars)
    n_rows = math.ceil(n / 3)

    for i in range(0, n, 3):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for j in range(3):
            if i + j < n:
                col = num_vars[i + j]
                sns.histplot(data[col].dropna(), kde=True, bins=30, color='steelblue', ax=axs[j])
                axs[j].set_title(f'{col}')
            else:
                axs[j].axis('off')
        plt.tight_layout()
        plt.show()

# ------------------------------
# Bivariada - Categóricas x Target
# ------------------------------
def bivariate_categorical_target(data, cat_vars, target):
    n = len(cat_vars)
    for i in range(0, n, 3):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for j in range(3):
            if i + j < n:
                col = cat_vars[i + j]
                prop_df = pd.crosstab(data[col], data[target], normalize='index')
                prop_df.plot(kind='bar', stacked=True, colormap='Set2', ax=axs[j], legend=False)
                axs[j].set_title(f'{col} vs {target}')
                axs[j].tick_params(axis='x', rotation=45)
                axs[j].set_ylabel('Proporção')
            else:
                axs[j].axis('off')
        plt.tight_layout()
        plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

# ------------------------------
# Bivariada - Numéricas x Target
# ------------------------------
def bivariate_numerical_target(data, num_vars, target):
    n = len(num_vars)
    for i in range(0, n, 3):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for j in range(3):
            if i + j < n:
                col = num_vars[i + j]
                sns.boxplot(data=data, x=target, y=col, palette='pastel', ax=axs[j])
                axs[j].set_title(f'{col} por {target}')
                axs[j].tick_params(axis='x', rotation=45)
            else:
                axs[j].axis('off')
        plt.tight_layout()
        plt.show()
        
def calculate_metrics(train_df, test_df, score_column, target_column,bins=10):
    def compute_metrics(df, score_column, target_column):
        df_sorted = df.sort_values(by=score_column, ascending=False)

        # Calcular KS
        df_sorted['cum_good'] = (1 - df_sorted[target_column]).cumsum() / (1 - df_sorted[target_column]).sum()
        df_sorted['cum_bad'] = df_sorted[target_column].cumsum() / df_sorted[target_column].sum()
        df_sorted['ks'] = np.abs(df_sorted['cum_good'] - df_sorted['cum_bad'])
        ks_statistic = df_sorted['ks'].max()

        # Calcular AUC
        auc_value = roc_auc_score(df_sorted[target_column], df_sorted[score_column])

        # Calcular Gini
        gini = 2 * auc_value - 1

        # Dividir o score em 10 faixas
        df_sorted['decile'] = pd.cut(df_sorted[score_column], bins, labels=False)

        # Criar tabela detalhada
        table = df_sorted.groupby('decile').agg(
            min_score=pd.NamedAgg(column=score_column, aggfunc='min'),
            max_score=pd.NamedAgg(column=score_column, aggfunc='max'),
            event_rate=pd.NamedAgg(column=target_column, aggfunc='mean'),
            volume=pd.NamedAgg(column=target_column, aggfunc='size')
        ).reset_index()

        return ks_statistic, auc_value, gini, table

    ks_train, auc_train, gini_train, table_train = compute_metrics(train_df, score_column, target_column)
    ks_test, auc_test, gini_test, table_test = compute_metrics(test_df, score_column, target_column)

    # Plotando o gráfico de barras para Event Rate por Decil
    barWidth = 0.3
    r1 = np.arange(len(table_train))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, table_train['event_rate'], color='lightblue', width=barWidth, label='Train')
    plt.bar(r2, table_test['event_rate'], color='royalblue', width=barWidth, label='Test')

    plt.xlabel('Decile')
    plt.ylabel('Event Rate')
    plt.title('Event Rate by Decile')
    plt.xticks([r + barWidth for r in range(len(table_train))], table_train['decile'])
    plt.legend()
    plt.show()

    # Criando DataFrame para as métricas
    metrics_df = pd.DataFrame({
        'Metric': ['KS', 'AUC', 'Gini'],
        'Train Value': [ks_train, auc_train, gini_train],
        'Test Value': [ks_test, auc_test, gini_test]
    })

    return metrics_df, table_train, table_test
  
def rename_features_as_var(df, target_col='TARGET'):
    """
    Renomeia todas as colunas preditoras como VAR_1, VAR_2, ..., mantendo o nome da variável target.
    Retorna o DataFrame renomeado e um DataFrame de-para com os nomes antigos e novos.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nome da variável alvo (não será renomeada).

    Retorna:
        df_renamed (pd.DataFrame): DataFrame com as variáveis renomeadas.
        depara_df (pd.DataFrame): Tabela de correspondência (de-para) entre nomes antigos e novos.
    """
    new_columns = {}
    depara = []
    count = 1

    for col in df.columns:
        if col != target_col:
            new_name = f'VAR_{count}'
            new_columns[col] = new_name
            depara.append({'Original Name': col, 'New Name': new_name})
            count += 1

    df_renamed = df.rename(columns=new_columns)
    depara_df = pd.DataFrame(depara)

    return df_renamed, depara_df
  

  
def logistic_regression_with_scorecard(data, target_var, features):
    """
    Ajusta uma regressão logística e gera um scorecard com os coeficientes beta, p-valores e estatísticas de Wald.

    Parâmetros:
        data (pd.DataFrame): dataset com as features e target
        target_var (str): nome da variável target binária
        features (list): lista de variáveis explicativas (não deve conter o target!)

    Retorna:
        model: modelo ajustado do statsmodels
        scorecard: DataFrame com Beta, P-Value e Estatística de Wald ordenados
    """
    # Remove o target da lista de features se estiver presente
    features = [f for f in features if f != target_var]

    # Adiciona a constante para o intercepto
    data = data.copy()
    data = sm.add_constant(data)

    # Ajusta o modelo de regressão logística
    model = sm.Logit(data[target_var], data[features + ['const']]).fit(disp=0)

    # Extrai os resultados
    summary = model.summary2().tables[1]
    summary['Wald'] = summary['z']**2
    scorecard = summary[['Coef.', 'P>|z|', 'Wald']]
    scorecard.columns = ['Beta Coefficient', 'P-Value', 'Wald Statistic']
    scorecard = scorecard.sort_values(by='Wald Statistic', ascending=False)

    return model, scorecard
  
def one_hot_encode_and_save_encoder(df, target, encoder_path='../artifacts/onehot_encoder_reg_log.pkl'):
    """
    Aplica one-hot encoding a variáveis categóricas, salva o encoder e retorna o DataFrame transformado
    com as dummies e as variáveis numéricas + target.
    """
    # Identifica variáveis categóricas e numéricas (excluindo o target)
    categorical_cols = [
        col for col in df.columns
        if col != target and (
            pd.api.types.is_object_dtype(df[col]) or
            pd.api.types.is_categorical_dtype(df[col])
        )
    ]
    numeric_cols = [
        col for col in df.columns
        if col != target and col not in categorical_cols
    ]

    # Instancia e aplica o OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])

    # Constrói DataFrame com os nomes das novas colunas
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Combina com as variáveis numéricas e o target
    final_df = pd.concat([df[numeric_cols], encoded_df, df[[target]]], axis=1)

    # Salva o encoder como .pkl
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    return final_df
  
def apply_saved_onehot_encoder(new_df, encoder_path='../artifacts/onehot_encoder.pkl', target=None):
    """
    Aplica um OneHotEncoder salvo via pickle a um novo DataFrame e retorna apenas as dummies
    e, opcionalmente, a variável target se fornecida.

    Parâmetros:
        new_df (pd.DataFrame): Novo DataFrame contendo as colunas categóricas esperadas.
        encoder_path (str): Caminho do arquivo .pkl do encoder salvo.
        target (str, opcional): Nome da variável target a ser mantida no retorno.

    Retorna:
        pd.DataFrame: DataFrame com as variáveis dummificadas e o target (se fornecido).
    """
    # Carrega o encoder salvo
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Recupera as colunas que o encoder espera
    expected_categorical_cols = encoder.feature_names_in_

    # Verifica se todas as colunas necessárias estão presentes
    missing_cols = set(expected_categorical_cols) - set(new_df.columns)
    if missing_cols:
        raise ValueError(f"As colunas esperadas pelo encoder estão ausentes no novo DataFrame: {missing_cols}")

    # Aplica o encoder
    encoded_array = encoder.transform(new_df[expected_categorical_cols])

    # Cria DataFrame com os nomes das colunas dummificadas
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(),
        index=new_df.index
    )

    # Adiciona o target, se solicitado e presente
    if target and target in new_df.columns:
        encoded_df[target] = new_df[target]

    return encoded_df
  
def plot_event_rate_barplots(df, target, ncols=4, max_unique=20):
    """
    Plota gráficos de barras em grid com a taxa do evento (média do target) para variáveis categóricas automaticamente detectadas.

    Parâmetros:
        df (pd.DataFrame): Dados de entrada
        target (str): Nome da variável target binária
        ncols (int): Número de colunas no grid de gráficos
        max_unique (int): Máximo de valores únicos para considerar uma variável como categórica (se for numérica)
    """
    # Detecta variáveis categóricas automaticamente
    categorical_vars = [
        col for col in df.columns
        if col != target and (
            pd.api.types.is_object_dtype(df[col]) or 
            pd.api.types.is_categorical_dtype(df[col]) or 
            (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= max_unique)
        )
    ]

    n_vars = len(categorical_vars)
    if n_vars == 0:
        print("Nenhuma variável categórica encontrada para plotar.")
        return

    nrows = math.ceil(n_vars / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, var in enumerate(categorical_vars):
        ax = axes[i]
        temp = df[[var, target]].dropna()
        event_rate = temp.groupby(var)[target].mean().sort_values(ascending=False)
        event_rate.plot(kind='bar', ax=ax)
        ax.set_title(f'{var} - Taxa do Evento')
        ax.set_xlabel(var)
        ax.set_ylabel('Taxa do Evento')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    # Esconde subplots extras, se houver
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def categorize_numerical_variables(df, bins=10, prefix='BIN_', target=None):
    """
    Categoriza todas as variáveis numéricas de um DataFrame, exceto o target, e mantém as variáveis categóricas inalteradas.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        bins (int): Número de faixas (quantis ou intervalos) para categorizar.
        prefix (str): Prefixo para as novas colunas categorizadas.
        target (str): Nome da variável target a ser preservada (não categorizada).

    Retorna:
        pd.DataFrame: Novo DataFrame com variáveis numéricas categorizadas (exceto o target) e categóricas preservadas.
    """
    df_transformed = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col == target:
            df_transformed[col] = df[col]
        elif pd.api.types.is_numeric_dtype(df[col]):
            try:
                df_transformed[f'{prefix}{col}'] = pd.qcut(df[col], q=bins, duplicates='drop')
            except ValueError:
                df_transformed[f'{prefix}{col}'] = pd.cut(df[col], bins=bins)
        else:
            df_transformed[col] = df[col]

    return df_transformed
  
  
def calculate_r2_for_logodds(df, variables, target, threshold):
    results = []

    for variable in variables:
        # Verificando o número de valores únicos
        unique_vals = df[variable].nunique()
        if unique_vals == 1:
            print(f"{variable} tem apenas um valor único. Ignorando...")
            continue

        n_bins = min(10, unique_vals)

        # Criando bins para a variável
        df['bin'] = pd.cut(df[variable], bins=n_bins, labels=False, duplicates='drop')

        # Calculando a proporção de eventos positivos para cada bin
        mean_target = df.groupby('bin')[target].mean()

        # Calculando o log(odds) e tratando valores infinitos
        log_odds = np.log(mean_target / (1 - mean_target)).replace([np.inf, -np.inf], np.nan).dropna()

        # Calculando R^2
        X = df.groupby('bin')[variable].mean()[log_odds.index].values.reshape(-1, 1)
        y = log_odds.values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        # Decidindo sobre a engenharia de recursos com base no valor de R^2 e no threshold fornecido
        feat_eng = "Usar como contínua" if r2 > threshold else "Categorizar"

        results.append({
            'Variable': variable,
            'R^2': r2,
            'Feat Eng': feat_eng
        })

        # Removendo a coluna bin
        df.drop('bin', axis=1, inplace=True)

    return pd.DataFrame(results)




def calculate_r2_for_logodds_with_grid(df, variables, target, threshold=0.85, ncols=4):
    results = []
    n_vars = len(variables)
    nrows = math.ceil(n_vars / ncols)
    eps = 1e-6

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, variable in enumerate(variables):
        ax = axes[i]
        unique_vals = df[variable].nunique()
        if unique_vals <= 1:
            ax.set_title(f'{variable}: único valor')
            ax.axis('off')
            continue

        df_temp = df[[variable, target]].copy()

        if pd.api.types.is_numeric_dtype(df_temp[variable]):
            n_bins = min(10, unique_vals)
            df_temp['bin'] = pd.cut(df_temp[variable], bins=n_bins, duplicates='drop')
            group_col = 'bin'
        else:
            df_temp['bin'] = df_temp[variable]
            group_col = 'bin'

        mean_target = df_temp.groupby(group_col)[target].mean().clip(eps, 1 - eps)
        log_odds = np.log(mean_target / (1 - mean_target)).dropna()

        if len(log_odds) < 2:
            ax.set_title(f'{variable}: poucos grupos válidos')
            ax.axis('off')
            continue

        if pd.api.types.is_numeric_dtype(df_temp[variable]):
            x_vals = df_temp.groupby('bin')[variable].mean()[log_odds.index].values.reshape(-1, 1)
        else:
            x_vals = np.arange(len(log_odds)).reshape(-1, 1)

        y_vals = log_odds.values

        model = LinearRegression().fit(x_vals, y_vals)
        r2 = model.score(x_vals, y_vals)
        feat_eng = "Usar como contínua" if r2 > threshold else "Categorizar"

        results.append({
            'Variable': variable,
            'R^2': r2,
            'Feat Eng': feat_eng
        })

        # Plot
        ax.plot(x_vals, y_vals, marker='o')
        ax.set_title(f'{variable} (R² = {r2:.2f})')
        ax.set_xlabel(variable)
        ax.set_ylabel('Log(Odds)')
        ax.grid(True)

    # Esconde gráficos extras
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

def selecionar_features_nulos(df, corte_pct_nulos=70):
    """
    Retorna uma lista com os nomes das variáveis cujo percentual de valores nulos
    está acima do valor de corte especificado.

    Parâmetros:
    - df: DataFrame contendo as colunas 'FEATURE' e 'PC_NULOS'
    - corte_pct_nulos: valor de corte percentual (ex: 95 significa 95%)

    Retorno:
    - Lista de nomes de features com % de nulos acima do corte
    """
    variaveis_descartadas = df[df['PC_NULOS'] > corte_pct_nulos]['FEATURE'].tolist()
    return variaveis_descartadas

def plot_variable_vs_logodds_grid(df, variables, target, ncols=4):
    n_vars = len(variables)
    nrows = math.ceil(n_vars / ncols)
    eps = 1e-6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  # facilita indexação

    for i, variable in enumerate(variables):
        ax = axes[i]
        unique_vals = df[variable].nunique()
        if unique_vals <= 1:
            ax.set_title(f'{variable}: único valor')
            ax.axis('off')
            continue

        df_temp = df[[variable, target]].copy()

        if pd.api.types.is_numeric_dtype(df_temp[variable]):
            n_bins = min(10, unique_vals)
            df_temp['bin'] = pd.cut(df_temp[variable], bins=n_bins, duplicates='drop')
            group_col = 'bin'
        else:
            df_temp['bin'] = df_temp[variable]
            group_col = 'bin'

        mean_target = df_temp.groupby(group_col)[target].mean()
        mean_target = mean_target.clip(eps, 1 - eps)
        log_odds = np.log(mean_target / (1 - mean_target)).replace([np.inf, -np.inf], np.nan).dropna()

        if len(log_odds) < 2:
            ax.set_title(f'{variable}: poucos grupos válidos')
            ax.axis('off')
            continue

        if pd.api.types.is_numeric_dtype(df_temp[variable]):
            x_vals = df_temp.groupby('bin')[variable].mean()[log_odds.index].values.reshape(-1, 1)
        else:
            x_vals = np.arange(len(log_odds)).reshape(-1, 1)

        y_vals = log_odds.values

        model = LinearRegression().fit(x_vals, y_vals)
        r2 = model.score(x_vals, y_vals)

        ax.plot(x_vals, y_vals, marker='o')
        ax.set_title(f'{variable} (R² = {r2:.2f})')
        ax.set_xlabel(variable)
        ax.set_ylabel('Log(Odds)')
        ax.grid(True)

    # Esconde subplots extras (se houver)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_variable_vs_logodds(df, variables, target):
    for variable in variables:
        # Verificando o número de valores únicos
        unique_vals = df[variable].nunique()
        if unique_vals == 1:
            print(f"{variable} tem apenas um valor único. Ignorando...")
            continue

        n_bins = min(10, unique_vals)

        # Criando bins para a variável
        df['bin'] = pd.cut(df[variable], bins=n_bins, labels=False, duplicates='drop')

        # Calculando a proporção de eventos positivos para cada bin
        mean_target = df.groupby('bin')[target].mean()

        # Calculando o log(odds) e tratando valores infinitos
        log_odds = np.log(mean_target / (1 - mean_target)).replace([np.inf, -np.inf], np.nan).dropna()

        # Calculando R^2
        X = df.groupby('bin')[variable].mean()[log_odds.index].values.reshape(-1, 1)
        y = log_odds.values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        # Plotando
        plt.figure(figsize=(8, 6))
        plt.plot(X, y, marker='o')
        plt.xlabel(variable)
        plt.ylabel('Log(Odds)')
        plt.title(f'{variable} vs Log(Odds) of {target}\nR^2 = {r2:.2f}')
        plt.grid(True)
        plt.show()

        # Removendo a coluna bin
        df.drop('bin', axis=1, inplace=True)    
    
def logistic_regression_with_scorecard_2(data, target_var, features):
    """
    Ajusta uma regressão logística e gera um scorecard com os coeficientes beta, p-valores e estatísticas de Wald.

    Parâmetros:
        data (pd.DataFrame): dataset com as features e target
        target_var (str): nome da variável target binária
        features (list): lista de variáveis explicativas (não deve conter o target!)

    Retorna:
        model: modelo ajustado do statsmodels
        scorecard: DataFrame com Beta, P-Value e Estatística de Wald ordenados
    """
    # Remove o target da lista de features se estiver presente
    features = [f for f in features if f != target_var]
    
    # Verifica se todas as features existem no DataFrame
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"As seguintes features não existem no DataFrame: {missing_features}")
    
    # Cria cópia para não modificar o original
    data = data.copy()
    
    # Prepara a matriz de features (X) e target (y)
    X = data[features]
    y = data[target_var]
    
    # Adiciona a constante para o intercepto (forma correta)
    X = sm.add_constant(X, has_constant='add')  # Garante que a constante será adicionada
    
    # Ajusta o modelo de regressão logística
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        raise ValueError(f"Erro ao ajustar o modelo: {str(e)}")
    
    # Extrai os resultados
    summary = model.summary2().tables[1]
    summary['Wald'] = summary['z']**2
    scorecard = summary[['Coef.', 'P>|z|', 'Wald']]
    scorecard.columns = ['Beta Coefficient', 'P-Value', 'Wald Statistic']
    scorecard = scorecard.sort_values(by='Wald Statistic', ascending=False)

    return model, scorecard


################# 1 FEATURE SELECTION ##################

def pod_academy_generate_metadata(dataframe):
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'qt_nulos': dataframe.isnull().sum(),
        'percent_nulos': round((dataframe.isnull().sum() / len(dataframe)) * 100, 2),
        'cardinalidade': dataframe.nunique(),
    }).sort_values(by='percent_nulos', ascending=False).reset_index(drop=True)
    return metadata

def preprocessar_df(df, y_train, threshold=0.5, percentual_preenchimento=70):
    # Etapa 0: Remover colunas com percentual de nulos maior que o permitido
    metadata_df = pod_academy_generate_metadata(df)
    colunas_validas = metadata_df[metadata_df['percent_nulos'] <= percentual_preenchimento]['nome_variavel'].tolist()
    df = df[colunas_validas]

    # Separar variáveis numéricas e categóricas
    num_features = df.select_dtypes(exclude='object')
    cat_features = df.select_dtypes(include='object')

    lista_nums = num_features.columns
    lista_cats = cat_features.columns

    # Imputação numérica
    imputer_num = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer_num.fit_transform(num_features), columns=lista_nums, index=num_features.index)

    # Imputação categórica
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cat = pd.DataFrame(imputer_cat.fit_transform(cat_features), columns=lista_cats, index=cat_features.index)

    # Target Encoding
    ce = TargetEncoder()
    df_cat_encoded = ce.fit_transform(df_cat, y_train)

    # Concatenar numéricas e categóricas
    df_processed = pd.concat((df_num, df_cat_encoded), axis=1)

    # Remoção de variáveis altamente correlacionadas
    corr_matrix = df_processed.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    df_final = df_processed.drop(columns=to_drop)

    return df_final, to_drop, metadata_df

def preprocessar_df_skl(df, y_train, threshold=0.5, percentual_preenchimento=70, variancia_minima=0.1, tamanho_amostra=85000):
    # 🔹 Amostragem dos dados (se necessário)
    if tamanho_amostra is not None and len(df) > tamanho_amostra:
        df = df.sample(n=tamanho_amostra, random_state=42)
        y_train = y_train.loc[df.index]  # Garante alinhamento com X

    # Etapa 0: Remover colunas com percentual de nulos maior que o permitido
    metadata_df = pod_academy_generate_metadata(df)
    colunas_validas = metadata_df[metadata_df['percent_nulos'] <= percentual_preenchimento]['nome_variavel'].tolist()
    df = df[colunas_validas]

    # Separar variáveis numéricas e categóricas
    num_features = df.select_dtypes(exclude='object')
    cat_features = df.select_dtypes(include='object')

    lista_nums = num_features.columns
    lista_cats = cat_features.columns

    # Imputação numérica
    imputer_num = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer_num.fit_transform(num_features), columns=lista_nums, index=num_features.index)

    # Imputação categórica
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cat = pd.DataFrame(imputer_cat.fit_transform(cat_features), columns=lista_cats, index=cat_features.index)

    # Target Encoding
    ce = TargetEncoder()
    df_cat_encoded = ce.fit_transform(df_cat, y_train)

    # Concatenar numéricas e categóricas
    df_processed = pd.concat((df_num, df_cat_encoded), axis=1)

    # Remover variáveis com baixa variância
    variancias = df_processed.var()
    colunas_baixa_variancia = variancias[variancias < variancia_minima].index.tolist()
    df_processed = df_processed.drop(columns=colunas_baixa_variancia)

    # Remoção de variáveis altamente correlacionadas
    corr_matrix = df_processed.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    df_final = df_processed.drop(columns=to_drop)

    # Juntar todas as colunas removidas
    colunas_removidas = list(set(colunas_baixa_variancia + to_drop))

    return df_final, colunas_removidas, metadata_df


# Metadados referente ao conjunto de dados
def generate_metadata_v2(dataframe):
    """
    Gera um dataframe contendo metadados das colunas do dataframe fornecido.

    :param dataframe: DataFrame para o qual os metadados serão gerados.
    :return: DataFrame contendo metadados.
    """

    # Coleta de metadados básicos
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'qt_nulos': dataframe.isnull().sum(),
        'percent_nulos': round((dataframe.isnull().sum() / len(dataframe))* 100,2),
        'cardinalidade': dataframe.nunique(),
    })
    metadata=metadata.sort_values(by='percent_nulos',ascending=False)
    metadata = metadata.reset_index(drop=True)

    return metadata

def calculate_metrics_rl(train_df, test_df, score_column, score_0, target_column,bins=10):
    def compute_metrics(df, score_column, score_0, target_column):
        df_sorted = df.sort_values(by=score_column, ascending=False)

        # Calcular KS
        df_sorted['cum_good'] = (1 - df_sorted[target_column]).cumsum() / (1 - df_sorted[target_column]).sum()
        df_sorted['cum_bad'] = df_sorted[target_column].cumsum() / df_sorted[target_column].sum()
        df_sorted['ks'] = np.abs(df_sorted['cum_good'] - df_sorted['cum_bad'])
        ks_statistic = df_sorted['ks'].max()

        # Calcular AUC
        auc_value = roc_auc_score(df_sorted[target_column], df_sorted[score_column])

        # Calcular Gini
        gini = 2 * auc_value - 1

        # Dividir o score em 10 faixas
        df_sorted['decile'] = pd.qcut(df_sorted[score_0], bins, labels=False)

        # Criar tabela detalhada
        table = df_sorted.groupby('decile').agg(
            min_score=pd.NamedAgg(column=score_0, aggfunc='min'),
            max_score=pd.NamedAgg(column=score_0, aggfunc='max'),
            event_rate=pd.NamedAgg(column=target_column, aggfunc='mean'),
            volume=pd.NamedAgg(column=target_column, aggfunc='size')
        ).reset_index()

        return ks_statistic, auc_value, gini, table

    ks_train, auc_train, gini_train, table_train = compute_metrics(train_df, score_column, score_0, target_column)
    ks_test, auc_test, gini_test, table_test = compute_metrics(test_df, score_column, score_0, target_column)

    # Plotando o gráfico de barras para Event Rate por Decil
    barWidth = 0.3
    r1 = np.arange(len(table_train))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, table_train['event_rate'], color='lightblue', width=barWidth, label='Train')
    plt.bar(r2, table_test['event_rate'], color='royalblue', width=barWidth, label='Test')

    plt.xlabel('Decile')
    plt.ylabel('Event Rate')
    plt.title('Event Rate by Decile')
    plt.xticks([r + barWidth for r in range(len(table_train))], table_train['decile'])
    plt.legend()
    plt.show()

    # Criando DataFrame para as métricas
    metrics_df = pd.DataFrame({
        'Metric': ['KS', 'AUC', 'Gini'],
        'Train Value': [ks_train, auc_train, gini_train],
        'Test Value': [ks_test, auc_test, gini_test]
    })

    return metrics_df, table_train, table_test

def calculate_metrics_skl(train_df, test_df, score_column, score_0, target_column,bins=10):
    def compute_metrics(df, score_column, score_0, target_column):
        df_sorted = df.sort_values(by=score_column, ascending=False)

        # Calcular KS
        df_sorted['cum_good'] = (1 - df_sorted[target_column]).cumsum() / (1 - df_sorted[target_column]).sum()
        df_sorted['cum_bad'] = df_sorted[target_column].cumsum() / df_sorted[target_column].sum()
        df_sorted['ks'] = np.abs(df_sorted['cum_good'] - df_sorted['cum_bad'])
        ks_statistic = df_sorted['ks'].max()

        # Calcular AUC
        auc_value = roc_auc_score(df_sorted[target_column], df_sorted[score_column])

        # Calcular Gini
        gini = 2 * auc_value - 1

        # Dividir o score em 10 faixas
        df_sorted['decile'] = pd.qcut(df_sorted[score_0], bins, labels=False)

        # Criar tabela detalhada
        table = df_sorted.groupby('decile').agg(
            min_score=pd.NamedAgg(column=score_0, aggfunc='min'),
            max_score=pd.NamedAgg(column=score_0, aggfunc='max'),
            event_rate=pd.NamedAgg(column=target_column, aggfunc='mean'),
            volume=pd.NamedAgg(column=target_column, aggfunc='size')
        ).reset_index()

        return ks_statistic, auc_value, gini, table

    ks_train, auc_train, gini_train, table_train = compute_metrics(train_df, score_column, score_0, target_column)
    ks_test, auc_test, gini_test, table_test = compute_metrics(test_df, score_column, score_0, target_column)

    # Plotando o gráfico de barras para Event Rate por Decil
    barWidth = 0.3
    r1 = np.arange(len(table_train))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, table_train['event_rate'], color='lightblue', width=barWidth, label='Train')
    plt.bar(r2, table_test['event_rate'], color='royalblue', width=barWidth, label='Test')

    plt.xlabel('Decile')
    plt.ylabel('Event Rate')
    plt.title('Event Rate by Decile')
    plt.xticks([r + barWidth for r in range(len(table_train))], table_train['decile'])
    plt.legend()
    plt.show()

    # Criando DataFrame para as métricas
    metrics_df = pd.DataFrame({
        'Metric': ['KS', 'AUC', 'Gini'],
        'Train Value': [ks_train, auc_train, gini_train],
        'Test Value': [ks_test, auc_test, gini_test]
    })

    return metrics_df, table_train, table_test


def kdeplots_var_num_target(dataframe, target_column):
    """
    Plota gráficos kdeplot (Kernel Density Estimation) para todas as variáveis numéricas do DataFrame,
    discriminando as curvas de acordo com o valor da coluna target.

    :param dataframe: DataFrame contendo as variáveis numéricas e a coluna target.
    :param target_column: Nome da coluna target.
    """
    # Seleciona apenas colunas numéricas
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # Define o número de linhas com base no número de colunas numéricas
    nrows = len(numeric_columns) // 3 + (len(numeric_columns) % 3 > 0)

    # Inicializa o painel de gráficos
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(14, nrows * 4))

    # Ajusta o layout
    plt.tight_layout(pad=4)

    # Configura estilo e paleta de cores
    sns.set_style("whitegrid")

    # Plota kdeplots para cada coluna numérica, discriminando as curvas pelo valor da coluna target
    for i, column in enumerate(numeric_columns):
        sns.kdeplot(data=dataframe[dataframe[target_column] == 1][column], ax=axes[i//3, i%3], color="blue", label="1", fill=True, warn_singular=False)
        sns.kdeplot(data=dataframe[dataframe[target_column] == 0][column], ax=axes[i//3, i%3], color="red", label="0", fill=True, warn_singular=False)
        axes[i//3, i%3].set_title(f'{column}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        axes[i//3, i%3].set_ylabel('Densidade')
        axes[i//3, i%3].tick_params(axis='both', which='major', labelsize=12)
        if i == 0:
            axes[i//3, i%3].legend(title=target_column)

    # Remove gráficos vazios (se houver)
    for j in range(i+1, nrows*3):
        fig.delaxes(axes.flatten()[j])

    # Adiciona título principal
    fig.suptitle("Análise descritiva - Gráfico KDE por Target", fontsize=20, fontweight='bold', y=1.00)
    
    
def pod_count_categorias(df, columns):
    if isinstance(columns, str):
        columns = [columns]  # Se apenas uma variável for passada, converte para lista

    # Calcula a contagem de valores e converte em DataFrame
    count_df = df[columns].value_counts().reset_index()
    count_df.columns = columns + ['Count']

    # Calcula a porcentagem de cada valor
    count_df['Percentage'] = (count_df['Count'] / count_df['Count'].sum()) * 100

    # Calcula a soma total da coluna de contagem
    total_count = count_df['Count'].sum()

    # Adiciona a soma total ao DataFrame
    total_row = pd.DataFrame({columns[0]: ['Total'], 'Count': [total_count], 'Percentage': [100]})
    count_df = pd.concat([count_df, total_row], ignore_index=True)

    return count_df





class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    Uma classe transformadora para remover colunas especificadas de um DataFrame.

    Atributos:
        to_drop (list): Uma lista de nomes de colunas a serem removidas.

    Métodos:
        fit(X, y=None): Ajusta o transformador aos dados. Este método não faz nada e existe apenas para conformidade com a API do Scikit-learn.
        transform(X): Transforma o DataFrame de entrada removendo as colunas especificadas.
    '''

    def __init__(self, to_drop):
        '''
        Inicializa o transformador ColumnDropper.

        Args:
            to_drop (list): Uma lista de nomes de colunas a serem removidas.
        '''
        self.to_drop = to_drop

    def fit(self, X, y=None):
        '''
        Ajusta o transformador aos dados.

        Este método não faz nada e existe apenas para conformidade com a API do Scikit-learn.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.
            y (array-like, default=None): Rótulos alvo. Ignorado.

        Returns:
            self: Retorna uma instância de self.
        '''
        return self

    def transform(self, X):
        '''
        Transforma o DataFrame de entrada removendo as colunas especificadas.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.

        Returns:
            pandas.DataFrame: DataFrame transformado após a remoção das colunas especificadas.
        '''
        # Garante que apenas colunas presentes serão removidas.
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        
        # Remove as colunas especificadas.
        return X.drop(columns=self.to_drop)
    

class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(
            drop='first',
            sparse_output=False,
            dtype=np.int8,
            handle_unknown='ignore'
        )
    
    def fit(self, X, y=None):
        # Se X for um DataFrame, salva os nomes das colunas
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            X_fit = X[self.to_encode]
        else:
            raise ValueError("OneHotFeatureEncoder espera um DataFrame como entrada no fit.")
        
        self.encoder.fit(X_fit)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("OneHotFeatureEncoder espera um DataFrame como entrada no transform.")
        
        X_one_hot = self.encoder.transform(X[self.to_encode])
        one_hot_df = pd.DataFrame(
            X_one_hot,
            columns=self.encoder.get_feature_names_out(self.to_encode),
            index=X.index  # mantém o índice original
        )
        
        return pd.concat([X.drop(columns=self.to_encode), one_hot_df], axis=1)

class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    '''
    Uma classe transformadora para padronização de características numéricas especificadas, mantendo os nomes das colunas.

    Atributos:
        to_scale (list): Uma lista de nomes de colunas a serem padronizadas.

    Métodos:
        fit(X, y=None): Ajusta o transformador aos dados.
        transform(X): Transforma o DataFrame de entrada padronizando as colunas especificadas e mantendo os nomes das colunas.
    '''
    def __init__(self, to_scale):
        '''
        Inicializa o transformador StandardFeatureScaler.

        Args:
            to_scale (list): Uma lista de nomes de colunas a serem padronizadas.
        '''
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        '''
        Ajusta o transformador aos dados.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.
            y (array-like, default=None): Rótulos alvo. Ignorado.

        Returns:
            self: Retorna uma instância de self.
        '''
        self.scaler.fit(X[self.to_scale])
        return self

    def transform(self, X):
        '''
        Transforma o DataFrame de entrada padronizando as colunas especificadas e mantendo os nomes das colunas.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.

        Returns:
            pandas.DataFrame: DataFrame transformado após a padronização das colunas especificadas, mantendo os nomes das colunas.
        '''
        # Padroniza as colunas.
        X_scaled = self.scaler.transform(X[self.to_scale])
        
        # Cria um dataframe para os dados padronizados.
        scaled_df = pd.DataFrame(X_scaled,
                                 columns=self.scaler.get_feature_names_out(self.to_scale))
        
        # Reseta os índices para mapeamento e concatena construindo um dataframe final de variáveis.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    Uma classe transformadora para codificação ordinal de características categóricas especificadas, mantendo os nomes das colunas.

    Atributos:
        to_encode (dict): Um dicionário onde as chaves são nomes de colunas e os valores são listas representando a ordem desejada das categorias.

    Métodos:
        fit(X, y=None): Ajusta o transformador aos dados.
        transform(X): Transforma o DataFrame de entrada aplicando codificação ordinal nas colunas especificadas e mantendo os nomes das colunas.
    '''
    def __init__(self, to_encode):
        '''
        Inicializa o transformador OrdinalFeatureEncoder.

        Args:
            to_encode (dict): Um dicionário onde as chaves são nomes de colunas e os valores são listas representando a ordem desejada das categorias.
        '''
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, 
                                      categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        '''
        Ajusta o transformador aos dados.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.
            y (array-like, default=None): Rótulos alvo. Ignorado.

        Returns:
            self: Retorna uma instância de self.
        '''
        self.encoder.fit(X[list(self.to_encode.keys())])
        return self

    def transform(self, X):
        '''
        Transforma o DataFrame de entrada aplicando codificação ordinal nas colunas especificadas e mantendo os nomes das colunas.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.

        Returns:
            pandas.DataFrame: DataFrame transformado após a codificação ordinal das colunas especificadas, mantendo os nomes das colunas.
        '''
        # Aplica codificação ordinal nas colunas.
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        
        # Cria um dataframe para os dados codificados ordinalmente.
        ordinal_encoded_df = pd.DataFrame(X_ordinal,
                                          columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        
        # Reseta os índices para mapeamento e concatena construindo um dataframe final de variáveis.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    

class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    Uma classe transformadora para codificação target de variáveis categóricas especificadas.

    Atributos:
        to_encode (list): Uma lista de nomes de colunas a serem codificadas por target.

    Métodos:
        fit(X, y=None): Ajusta o transformador aos dados.
        transform(X): Transforma o DataFrame de entrada aplicando codificação target nas colunas especificadas.
    '''

    def __init__(self, to_encode):
        '''
        Inicializa o transformador TargetFeatureEncoder.

        Args:
            to_encode (list): Uma lista de nomes de colunas a serem codificadas por target.
        '''
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        '''
        Ajusta o transformador aos dados.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.
            y (array-like): Rótulos alvo.

        Returns:
            self: Retorna uma instância de self.
        '''
        self.encoder.fit(X[self.to_encode], y)
        return self

    def transform(self, X):
        '''
        Transforma o DataFrame de entrada aplicando codificação target nas colunas especificadas.

        Args:
            X (pandas.DataFrame): Variáveis de entrada.

        Returns:
            pandas.DataFrame: DataFrame transformado após a codificação target das colunas especificadas.
        '''
        # Aplica codificação target nas colunas.
        X_target = self.encoder.transform(X[self.to_encode])

        # Cria um dataframe para os dados codificados por target.
        target_df = pd.DataFrame(X_target,
                                 columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reseta os índices para mapeamento e concatena construindo um dataframe final de variáveis.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)
    
    
def boxplots_var_num(dataframe):
    """
    Plota boxplots para todas as variáveis numéricas do dataframe fornecido em um painel com 3 gráficos por linha.

    :param dataframe: DataFrame para o qual os boxplots serão gerados.
    """
    # Seleciona apenas colunas numéricas
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # Define o número de linhas com base no número de colunas numéricas
    nrows = len(numeric_columns) // 3 + (len(numeric_columns) % 3 > 0)

    # Inicializa o painel de gráficos
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(14, nrows * 4))

    # Ajusta o layout
    plt.tight_layout(pad=4)

    # Configura estilo e paleta de cores
    sns.set_style("whitegrid")

    # Plota boxplots para cada coluna numérica
    for i, column in enumerate(numeric_columns):
        sns.boxplot(data=dataframe[column], ax=axes[i//3, i%3], color="skyblue")
        axes[i//3, i%3].set_title(f'{column}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        axes[i//3, i%3].set_ylabel('')

    # Remove gráficos vazios (se houver)
    for j in range(i+1, nrows*3):
        fig.delaxes(axes.flatten()[j])

    # Adiciona título principal
    fig.suptitle("Análise descritiva - Boxplots", fontsize=20, fontweight='bold', y=1.00)
    
    
def histograms_var_num(dataframe):
    """
    Plota histogramas corrigidos com a curva KDE (Kernel Density Estimation) para todas as variáveis numéricas
    do dataframe fornecido em um painel com 3 gráficos por linha.

    :param dataframe: DataFrame para o qual os histogramas serão gerados.
    """
    # Seleciona apenas colunas numéricas
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # Define o número de linhas com base no número de colunas numéricas
    nrows = len(numeric_columns) // 3 + (len(numeric_columns) % 3 > 0)

    # Inicializa o painel de gráficos
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(14, nrows * 4))

    # Ajusta o layout
    plt.tight_layout(pad=4)

    # Configura estilo e paleta de cores
    sns.set_style("whitegrid")

    # Plota histogramas com KDE para cada coluna numérica
    for i, column in enumerate(numeric_columns):
        sns.histplot(data=dataframe[column], ax=axes[i//3, i%3], color="skyblue", bins=30, kde=True)
        axes[i//3, i%3].set_title(f'{column}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        axes[i//3, i%3].set_ylabel('Frequência')
        axes[i//3, i%3].tick_params(axis='both', which='major', labelsize=12)

    # Remove gráficos vazios (se houver)
    for j in range(i+1, nrows*3):
        fig.delaxes(axes.flatten()[j])

    # Adiciona título principal
    fig.suptitle("Análise descritiva - Histograma com KDE", fontsize=20, fontweight='bold', y=1.00)
    
def plot_categorical_frequency_pt(df, corte_cardinalidade=30, graficos_por_linha=2):
    """
    Plota a frequência de categorias para variáveis categóricas em um DataFrame.

    Parâmetros:
    - df: DataFrame para plotagem.
    - corte_cardinalidade: Cardinalidade máxima para uma coluna ser considerada (padrão é 30).
    - graficos_por_linha: Quantidade de gráficos por linha (padrão é 3).

    Retorna:
    - Exibe os gráficos de barras.
    """

    # Gera metadados para o DataFrame
    metadados = []
    for coluna in df.columns:
        metadados.append({
            'Variável': coluna,
            'Tipo': df[coluna].dtype,
            'Cardinalidade': df[coluna].nunique()
        })

    df_metadados = pd.DataFrame(metadados)

    # Filtra colunas com cardinalidade maior que o corte e tipos não numéricos
    variaveis_categoricas = df_metadados[(df_metadados['Cardinalidade'] <= corte_cardinalidade) & (df_metadados['Tipo'] == 'object')]

    # Calcula o número de linhas e colunas para os subplots
    n_linhas = -(-len(variaveis_categoricas) // graficos_por_linha)  # Ceiling division
    n_colunas = min(len(variaveis_categoricas), graficos_por_linha)

    # Plota as variáveis categóricas
    fig, axs = plt.subplots(nrows=n_linhas, ncols=n_colunas, figsize=(12, 4 * n_linhas))

    for i, (idx, linha) in enumerate(variaveis_categoricas.iterrows()):
        var = linha['Variável']
        ax = axs[i // graficos_por_linha, i % graficos_por_linha]
        df[var].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'Frequência em {var}')
        ax.set_ylabel('Frequência')
        ax.set_xlabel(var)

    # Remove os eixos vazios, se houver
    for j in range(i + 1, n_linhas * n_colunas):
        axs[j // graficos_por_linha, j % graficos_por_linha].axis('off')

    plt.tight_layout()
    plt.show()
    
def kdeplots_var_num_target(dataframe, target_column):
    """
    Plota gráficos kdeplot (Kernel Density Estimation) para todas as variáveis numéricas do DataFrame,
    discriminando as curvas de acordo com o valor da coluna target.

    :param dataframe: DataFrame contendo as variáveis numéricas e a coluna target.
    :param target_column: Nome da coluna target.
    """
    # Seleciona apenas colunas numéricas
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # Define o número de linhas com base no número de colunas numéricas
    nrows = len(numeric_columns) // 3 + (len(numeric_columns) % 3 > 0)

    # Inicializa o painel de gráficos
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(14, nrows * 4))

    # Ajusta o layout
    plt.tight_layout(pad=4)

    # Configura estilo e paleta de cores
    sns.set_style("whitegrid")

    # Plota kdeplots para cada coluna numérica, discriminando as curvas pelo valor da coluna target
    for i, column in enumerate(numeric_columns):
        sns.kdeplot(data=dataframe[dataframe[target_column] == 1][column], ax=axes[i//3, i%3], color="blue", label="1", fill=True, warn_singular=False)
        sns.kdeplot(data=dataframe[dataframe[target_column] == 0][column], ax=axes[i//3, i%3], color="red", label="0", fill=True, warn_singular=False)
        axes[i//3, i%3].set_title(f'{column}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        axes[i//3, i%3].set_ylabel('Densidade')
        axes[i//3, i%3].tick_params(axis='both', which='major', labelsize=12)
        if i == 0:
            axes[i//3, i%3].legend(title=target_column)

    # Remove gráficos vazios (se houver)
    for j in range(i+1, nrows*3):
        fig.delaxes(axes.flatten()[j])

    # Adiciona título principal
    fig.suptitle("Análise descritiva - Gráfico KDE por Target", fontsize=20, fontweight='bold', y=1.00)
    
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cat_vs_target_percentage(dataframe, target_column, cutoff=10):
    """
    Plota gráficos de barras empilhadas 100% para analisar as variáveis categóricas em relação ao target,
    limitando o número de variáveis de acordo com um valor de cutoff.

    :param dataframe: DataFrame contendo as variáveis categóricas e a coluna target.
    :param target_column: Nome da coluna target.
    :param cutoff: Valor de cutoff para limitar o número de variáveis categóricas plotadas (padrão é 10).
    """
    # Seleciona apenas colunas categóricas
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns

    # Filtra as colunas com base no cutoff
    categorical_columns_filtered = [col for col in categorical_columns if dataframe[col].nunique() <= cutoff]

    # Define o número de linhas e colunas para os subplots
    n_rows = len(categorical_columns_filtered) // 3 + (len(categorical_columns_filtered) % 3 > 0)
    n_cols = min(len(categorical_columns_filtered), 3)

    # Cria subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))

    # Ajusta o layout
    plt.tight_layout(pad=4)

    # Loop pelas colunas categóricas filtradas
    for i, column in enumerate(categorical_columns_filtered):
        # Calcula proporções de cada categoria para cada valor do target
        prop_df = (dataframe.groupby([column, target_column]).size() / dataframe.groupby(column).size()).unstack()

        # Plota o gráfico de barras empilhadas 100%
        ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i]
        prop_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(column, fontsize=14)
        ax.set_ylabel('Porcentagem')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Rotaciona as labels do eixo x
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Ajusta o layout do subplot
        plt.subplots_adjust(wspace=0.5, hspace=0.7)

    # Remove subplots vazios
    for j in range(len(categorical_columns_filtered), n_rows * n_cols):
        if n_rows > 1:
            fig.delaxes(axes.flatten()[j])
        else:
            fig.delaxes(axes)

    # Adiciona título principal
    fig.suptitle("Análise de Variáveis Categóricas em relação ao Target (Porcentagem)", fontsize=20, fontweight='bold', y=1.02)