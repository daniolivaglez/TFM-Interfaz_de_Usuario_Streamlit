import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils.np_utils import to_categorical
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

image_path = "C:/Users/DEEPGAMING/Desktop/DANI/Máster Avanzado de Programación en Python para Hacking, BigData y Machine Learning IV/TFM/código/imagenes/"


class DataTFM():
    def tfm(self):
        self.home()
        self.prueba()

    def home(self):
        st.set_page_config(page_icon="📊", page_title="TFM de Daniel", layout="wide")
        # st.write(st.session_state)

        # "st session: ",  st.session_state
        st.image("https://www.codificandobits.com/img/cb-logo.png", width=200)
        st.title("Interfaz para predecir datos")
        self.container = st.container()

    def prueba(self):

        uploaded_file = self.container.file_uploader(
            "Añada el archivo .csv", type='csv',
            key="uploaded_file",
        )
        if uploaded_file:
            self.datos = pd.read_csv(uploaded_file, sep=";")

        else:
            self.container.info(
                f"""
                    👆 Debe cargar primero un dato con extensión .csv
                    """
            )
            st.stop()

        if "clicked_mostrar_datos" not in st.session_state and "selected_var_pred" not in st.session_state:
            st.session_state["clicked_mostrar_datos"] = False
            st.session_state["selected_var_pred"] = False

        if "clicked_realizar_predicciones" not in st.session_state:
            st.session_state["clicked_realizar_predicciones"] = False

        if "clicked_transformar_datos" not in st.session_state:
            st.session_state["clicked_transformar_datos"] = False

        self.c1, self.c2, self.c3 = st.columns([1,4,1])
        with self.c2:
            st.dataframe(self.datos.describe(), use_container_width=True)

        self.variables = list(self.datos.columns)

        self.variables_selector = [None] + self.variables

        variable_pred = st.selectbox('Seleccione la variable a predecir', self.variables_selector)
        self.variable_pred = variable_pred

        if variable_pred is not None:
            st.success('La variable para realizar la predicción seleccionada es: ' + variable_pred)
            st.markdown('Se va a hacer un plot de las diferentes clasificaciones')
            self.scatter(variable_pred)

        if st.button('Mostrar datos', key='mostrar_datos') or st.session_state["clicked_mostrar_datos"]:
            st.session_state["clicked"] = True
            st.session_state["clicked_mostrar_datos"] = True
            st.session_state["selected_var_pred"] = True

            self.lista = list(self.datos.columns)

            self.mostrar_columnas_datos()

            if st.button('Realizar predicciones', key='realizar_predicciones') or st.session_state[
                'clicked_realizar_predicciones']:
                st.session_state['clicked_realizar_predicciones'] = True
                self.mostrar_radio()

    def scatter(self, variable_pred):

        if 'str' in str(type(self.datos[variable_pred][0])):
            self.variables.remove(variable_pred)
            self.combinaciones_scatter = list(combinations(self.variables, 2))

            n_combinaciones = len(self.combinaciones_scatter)
            self.lista_images = []

            for i, combinacion in enumerate(self.combinaciones_scatter):
                g = sns.FacetGrid(self.datos, hue=variable_pred, height=3).map(plt.scatter, combinacion[0],
                                                                               combinacion[1]).add_legend()
                g.fig.suptitle(f'Clasificación de {self.variable_pred} según {combinacion[0]}-{combinacion[1]}',
                               fontsize=10)
                nombre_imagen = f"{image_path}{'_'.join((combinacion))}_{time.time()}.png"
                g.savefig(nombre_imagen)

                self.lista_images.append(nombre_imagen)

            fig = plt.figure(figsize=(10, 7))
            rows = 2
            columns = int(np.round(n_combinaciones / 2))

            for i, image_p in enumerate(self.lista_images):
                fig.add_subplot(rows, columns, i + 1)
                image = Image.open(image_p)
                plt.imshow(image)
                plt.axis('off')

            st.pyplot(fig)

        else:
            st.error('La variable seleccionada es numérica')

    def mostrar_columnas_datos(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('Datos originales')
            st.dataframe(self.datos)

            if st.button('Transformar datos', key='transformar_datos') or st.session_state["clicked_transformar_datos"]:
                st.session_state["clicked_transformar_datos"] = True
                self.transformar_datos(self.lista)

                with col2:
                    st.markdown('Datos X train')
                    st.dataframe(self.x_train)

                with col3:
                    st.markdown('Datos Y train')
                    st.dataframe(self.y_train)

    def transformar_datos(self, lista_variable):

        self.y_train_cat = self.datos[self.variable_pred]
        self.x_train = self.datos.drop(self.variable_pred, axis=1)
        self.diccionario_variables = {}
        self.diccionario_scaler = {}

        # st.write(list)

        for variable in lista_variable:
            if not self.variable_pred:
                st.error('Seleccione una variable a predecir')

            if variable == self.variable_pred:
                y = np.unique(self.y_train_cat)
                names = y
                self.clases = len(names)

                st.write("Hay", self.clases, "clases en el conjunto de datos")
                self.mapping = {key: value for key, value in zip(y, range(len(y)))}
                processed_y = np.array([self.mapping[i] for i in self.datos[variable]])
                self.y_train = to_categorical(processed_y, num_classes=self.clases)

            elif variable != self.variable_pred and 'str' in str(type(self.datos[variable][0])):
                y = np.unique(self.datos[variable])
                mapping = {key: value for key, value in zip(y, range(len(y)))}
                processed_y = np.array([mapping[i] for i in self.datos[variable]])

                self.x_train[variable] = processed_y
                self.diccionario_variables[variable] = mapping

            else:
                scaler = StandardScaler()
                self.x_train[variable] = scaler.fit_transform(pd.DataFrame(self.x_train[variable]))
                self.diccionario_variables[variable] = 'float'
                self.diccionario_scaler[variable] = scaler

    def mostrar_radio(self):
        opcion = st.radio('Seleccione el algoritmo para realizar la predicción',
                          ['None', 'Árbol de Decisión', 'SVM', 'KNN'],
                          key='selectbox_clasificador')
        if opcion == 'None':
            st.write('Debe seleccionar un clasificador')
        else:
            self.mostrar_inputs()
            if opcion == 'Árbol de Decisión':
                self.decision_tree_cl()
            elif opcion == 'SVM':
                self.svm()
            elif opcion == 'KNN':
                self.knn()

    def mostrar_inputs(self):
        self.valores = {}
        self.valores_escalados = {}
        st.write("Inserte los datos para realizar la predicción")

        for k, v in self.diccionario_variables.items():
            if v == 'float':
                valor = st.number_input(f"Inserte {k}: ")
                self.valores[k] = valor
                valor_escalado = self.diccionario_scaler[k].transform(pd.DataFrame([valor]))[0][0]
                self.valores_escalados[k] = valor_escalado

            else:
                opcion = st.selectbox(f'Escoja uno de los posibles valores de la variable {k}',
                                      list(self.diccionario_variables[k].keys()))
                self.valores[k] = self.diccionario_variables[k][opcion]
                st.write(self.valores[k])


    def decision_tree_cl(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.x_train, self.y_train)
        self.obtener_resultado(dt, 'tree_cl')

    def svm(self):
        svm = SVC()
        svm.fit(self.x_train, self.y_train_cat)
        self.obtener_resultado(svm, 'svm')

    def knn(self):
        n_vecinos = None
        n_vecinos = int(st.number_input('Introduzca el número de vecinos (número entero): ', value=self.clases))
        if n_vecinos:
            knn = KNeighborsClassifier(n_neighbors=n_vecinos)
            knn.fit(self.x_train, self.y_train)
            self.obtener_resultado(knn, 'knn')

    def obtener_resultado(self, clf, nombre_clf):
        self.resultado = None
        st.session_state['clicked_predecir_resultado'] = False
        if st.button('Predecir resultado', key='predecir_resultado') or st.session_state['clicked_predecir_resultado']:
            st.session_state['clicked_predecir_resultado'] = True

            prediccion = clf.predict([list(self.valores_escalados.values())])


            if nombre_clf == 'tree_cl' or nombre_clf == 'knn':
                self.resultado = self.key_de_value(self.mapping, np.argmax(prediccion))
            elif nombre_clf == 'svm':
                self.resultado = prediccion[0]

            st.write(f"El resultado de la predicción introducida es: {self.resultado}")


    def key_de_value(self, diccionario, valor):
        keys = [k for k, v in diccionario.items() if v == valor]
        if keys:
            return keys[0]
        return None


if __name__ == '__main__':
    data = DataTFM()
    data.tfm()