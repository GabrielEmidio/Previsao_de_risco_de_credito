import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.write("""
    Prevendo risco de crédito
""")

#dataset
base = pd.read_csv('C:\\Users\\gabri\\OneDrive\\Documentos\\ProjetoPython\\credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92

#Cabeçalho
st.subheader('Informações dos dados')

#Nome do usuário
user_input = st.sidebar.text_input('Digite seu nome')

st.write('Cliente: ', user_input)

#Separando os previsores e a classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Pré-processamento
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])


#scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)


#Separando treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#dados dos usuários com a função
def get_user_data():
    income = st.sidebar.slider('Renda (anual)', 0, 70000, 0)
    age = st.sidebar.slider('Idade', 18, 120, 18)
    loan = st.sidebar.slider('Empréstimo', 0, 13800, 0)

    user_data = {
        'Renda': income,
        'Idade': age,
        'Empréstimo': loan
    }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

#Gráfico
graf = st.table(user_input_variables)

st.subheader('Dados do cliente')
st.write(user_input_variables)

# Árvore de decisão
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Acuracia do modelo
st.subheader('Acurácia do modelo (%):')
st.write(accuracy_score(classe_teste, previsoes) * 100)

# Previsão
resultado = classificador.predict(user_input_variables)

st.subheader('Previsão: ')
st.write(resultado)


