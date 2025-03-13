import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("train.csv")

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualization", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("### Introduction")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA"):
        st.dataframe(df.isna().sum())

if page == pages[1]:
    st.write("### DataVizualization")

    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)

    fig = sns.displot(x='Age', data=df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig)

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    numeric_df = df.select_dtypes(include=['number'])

    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), ax=ax, annot=True, cmap='coolwarm')
        st.pyplot(fig)
    else:
        st.write("No numeric columns found in the dataset.")

if page == pages[2]:
    st.write("### Modélisation")

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex', 'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    clf = prediction(option)

    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

    # Import du classifieur clf précédemment entrainé et enregistré avec Joblib
    try:
        loaded_model_joblib = joblib.load("model")
        #st.write("Modèle Joblib chargé avec succès.")
    except FileNotFoundError:
        st.write("Le modèle Joblib n'a pas été trouvé.")

    # Import du classifieur clf précédemment entrainé et enregistré avec Pickle
    try:
        loaded_model_pickle = pickle.load(open("model", 'rb'))
        #st.write("Modèle Pickle chargé avec succès.")
    except FileNotFoundError:
        st.write("Le modèle Pickle n'a pas été trouvé.")

