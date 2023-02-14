import streamlit as st
import pandas as pd
import numpy as np
import time
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils.np_utils import to_categorical
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

image_path = "C:/Users/DEEPGAMING/Desktop/DANI/Máster Avanzado de Programación en Python para Hacking, BigData y Machine Learning IV/TFM/código/imagenes/"


class DataTFM():
    def tfm(self):
        self.home()


    def home(self):
        st.set_page_config(page_icon="📊", page_title="TFM de Daniel", layout="wide")
        # st.write(st.session_state)

        # "st session: ",  st.session_state
        st.image("https://www.codificandobits.com/img/cb-logo.png", width=200)
        st.title("Interfaz para predecir datos")
        self.container = st.container()

        uploaded_file = self.container.file_uploader(
            "_Añada el archivo .csv_", type='csv',
            key="uploaded_file",
        )
        if uploaded_file:
            self.datos = pd.read_csv(uploaded_file, sep=";")

        else:
            self.container.info(
                f"""👆 Debe cargar primero un dato con extensión .csv"""
            )
            st.stop()

        if self.datos is not None:
            self.seleccionar_prediccion()

    def seleccionar_prediccion(self):
        self.tipo = st.radio('Seleccione si desea hacer una clasificación o una regresión', ['None', 'Clasificación', 'Regresión'], key='selectbox_tipo')
        if self.tipo:
            self.clasificar_variable()
        else:
            st.warning('Seleccione el tipo de predicción a realizar')

    def init_buttons(self):
        self.variables = list(self.datos.columns)
        self.contador_nulos = 0

        if "clicked_mostrar_datos" not in st.session_state and "selected_var_pred" not in st.session_state:
            st.session_state["clicked_mostrar_datos"] = False
            st.session_state["selected_var_pred"] = False

        if "clicked_transformar_datos" not in st.session_state:
            st.session_state["clicked_transformar_datos"] = False

        if "clicked_ver_graficas" not in st.session_state:
            st.session_state["clicked_ver_graficas"] = False

        if "clicked_realizar_predicciones" not in st.session_state:
            st.session_state["clicked_realizar_predicciones"] = False

        if "clicked_imputar_valores" not in st.session_state:
            st.session_state["clicked_imputar_valores"] = False

        self.describe_dataset()

        if self.contador_nulos:
            if st.button('Imputar valores', key='imputar_valores') or st.session_state["clicked_imputar_valores"]:
                st.session_state['clicked_imputar_valores'] = True
                self.impute_values()
                st.success('Los datos se han imputado correctamente')

        self.variables_selector = [None] + self.variables

        self.variable_pred = st.selectbox('_Seleccione la variable a predecir_', self.variables_selector)

        if self.tipo == 'Clasificación':
            self.cambio_tipo_variable_pred()

    def describe_dataset(self):
        self.c1, self.c2 = st.columns([3, 3])
        with self.c1:
            st.title("Valores nulos visualmente")
            fig, ax = plt.subplots()
            g = sns.heatmap(self.datos.isnull(), cbar=False, ax=ax)
            st.write(fig)

        with self.c2:
            st.title("Estadística descriptiva")
            st.dataframe(self.datos.describe(), use_container_width=True)

            datosnulos = dict(self.datos.isnull().sum())
            for k, v in datosnulos.items():
                if v > 0:
                    self.contador_nulos += 1
                    st.warning(f'Para la variable {k} hay {v} valores nulos')

    def impute_values(self):
        simple = SimpleImputer(strategy='most_frequent')
        self.datos = pd.DataFrame(simple.fit_transform(self.datos), columns=self.variables)

    def clasificar_variable(self):

        self.init_buttons()

        if self.variable_pred is not None:
            st.success('La variable para realizar la predicción seleccionada es: ' + self.variable_pred)
            if st.button('Mostrar datos', key='mostrar_datos') or st.session_state["clicked_mostrar_datos"]:
                st.session_state["clicked"] = True
                st.session_state["clicked_mostrar_datos"] = True
                st.session_state["selected_var_pred"] = True

                self.lista = list(self.datos.columns)

                if self.tipo:
                    self.mostrar_columnas_datos()
                else:
                    st.warning('Seleccione el tipo de predicción a realizar')

                if st.button('Ver gráficas', key='ver_graficas') or st.session_state["clicked_ver_graficas"]:
                    st.session_state['clicked_ver_graficas'] = True
                    if self.variable_pred in self.variables:
                        self.variables.remove(self.variable_pred)

                    opciones = st.multiselect(label='Seleccionar las diferentes variables para realizar las gráficas', options=self.variables)
                    self.combinaciones_graficas = list(combinations(opciones, 2))
                    n_combinaciones = len(self.combinaciones_graficas)

                    self.lista_images = []
                    self.rows = 2
                    self.columns = 1 if int(np.round(n_combinaciones / 2)) == 0 else int(np.round(n_combinaciones / 2))

                    st.markdown('Se va a hacer un plot de las diferentes clasificaciones')

                    if self.tipo == 'Clasificación':
                        self.graficas_clasificacion()

                    elif self.tipo == 'Regresión':
                        self.radio_seleccion_tipo_regresion()

                if st.button('Realizar predicciones', key='realizar_predicciones') or st.session_state[
                    'clicked_realizar_predicciones']:
                    st.session_state['clicked_realizar_predicciones'] = True

                    if self.tipo == 'Clasificación':
                        self.radio_seleccion_tipo_clasificador()

                    elif self.tipo =='Regresión':
                        self.radio_seleccion_tipo_regresion()

    def graficas_clasificacion(self):
        tab1, tab2, tab3 = st.tabs(["Matriz correlación", "FacetGrid", "Violinplot"])
        with tab1:
            self.matriz_correlation()

        with tab2:
            self.scatter()

        with tab3:
            self.violinplot()

    def cambio_tipo_variable_pred(self):
        if self.variable_pred:
            self.valores_unicos = np.unique(self.datos[self.variable_pred])

            if len(self.valores_unicos) == 2:
                for i in self.valores_unicos:
                    valor = st.text_input(f"Inserte valor para {i}: ")
                    self.datos[self.variable_pred] = self.datos[self.variable_pred].replace({i: valor})

    def violinplot(self):
        if 'str' in str(type(self.datos[self.variable_pred][0])):
            opcion_plot = st.selectbox('Seleccione la variable para mostrar el violinplot', self.variables)

            if opcion_plot:
                c1, c2, c3 = st.columns([2, 2, 2])

                with c2:
                    fig, ax = plt.subplots()
                    sns.violinplot(data=self.datos, x=opcion_plot, y=self.variable_pred, inner="stick", ax=ax)
                    st.write(fig)

        else:
            st.error('La variable seleccionada es numérica')

    def matriz_correlation(self):
        c1, c2, c3 = st.columns([2,2,2])
        with c2:
            fig, ax = plt.subplots()
            sns.heatmap(self.datos.corr(), cmap='coolwarm', ax=ax, annot=True, linewidth=.5)
            st.write(fig)

    def scatter(self):
        if 'str' in str(type(self.datos[self.variable_pred][0])):
            combinacion = st.selectbox('Seleccione la combinación para mostrar la gráfica', self.combinaciones_graficas)
            if combinacion:
                c1, c2, c3 = st.columns([2, 2, 2])
                with c2:
                    g = sns.FacetGrid(self.datos, hue=self.variable_pred, height=3).map(plt.scatter, combinacion[0],
                                                                                        combinacion[1]).add_legend()

                    g.fig.suptitle(f'Clasificación de {self.variable_pred} según {combinacion[0]}-{combinacion[1]}',
                                   fontsize=10)
                    st.pyplot(g)

        else:
            st.error('La variable seleccionada es numérica')

    def mostrar_columnas_datos(self):
        self.clases = None
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('Datos originales')
            st.dataframe(self.datos)

            opcion = st.selectbox('Seleccione cómo escalar los datos: ',
                                  [None, 'StandardScaler', 'MinMaxScaler', 'RobustScaler'])

            if opcion:
                porcentaje_test = float(st.number_input('Introduzca el procentaje de datos para test', value=0.2))
                if st.button('Transformar datos', key='transformar_datos') or st.session_state["clicked_transformar_datos"]:
                    st.session_state["clicked_transformar_datos"] = True

                    if opcion:
                        if self.tipo == 'Clasificación':
                            self.transformar_datos_clasificacion(self.lista, opcion)
                        elif self.tipo == 'Regresión':
                            self.transformar_datos_regresion(self.lista, opcion)

                        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.x_train, self.y_train, test_size=porcentaje_test)

                    else:
                        st.warning('Debe seleccionar un normalizador para las variables')

                    with c2:
                        st.markdown('Datos X train')
                        st.dataframe(self.X_train)

                        st.markdown('Datos X test')
                        st.dataframe(self.X_test)

                    with c3:
                        st.markdown('Datos Y train')
                        st.dataframe(self.Y_train)

                        st.markdown('Datos Y test')
                        st.dataframe(self.Y_test)

            else:
                st.warning('Debe seleccionar un normalizador para las variables')

        if self.clases:
            st.info(f"Hay {self.clases} clases en el conjunto de datos")

    def transformar_datos_clasificacion(self, lista_variable, opcion):
        self.y_train_cat = self.datos[self.variable_pred]
        self.x_train = self.datos.drop(self.variable_pred, axis=1)
        self.diccionario_variables = {}
        self.diccionario_scaler = {}

        for variable in lista_variable:
            if not self.variable_pred:
                st.error('Seleccione una variable a predecir')

            if variable == self.variable_pred:
                y = np.unique(self.y_train_cat)
                names = y
                self.clases = len(names)

                self.mapping = {key: value for key, value in zip(y, range(len(y)))}
                processed_y = np.array([self.mapping[i] for i in self.datos[variable]])
                self.y_train = to_categorical(processed_y, num_classes=self.clases)

            elif variable != self.variable_pred and 'str' in str(type(self.datos[variable][0])):
                self.mapeo_variable_tipo_string_a_int(variable)

            else:
                self.escalar_variable_int(opcion, variable)

    def mapeo_variable_tipo_string_a_int(self, variable):
        y = np.unique(self.datos[variable])
        mapping = {key: value for key, value in zip(y, range(len(y)))}
        processed_y = np.array([mapping[i] for i in self.datos[variable]])

        self.x_train[variable] = processed_y
        self.diccionario_variables[variable] = mapping

    def transformar_datos_regresion(self, lista_variable, opcion):
        self.y_train = self.datos[self.variable_pred]
        self.x_train = self.datos.drop(self.variable_pred, axis=1)
        self.diccionario_variables = {}
        self.diccionario_scaler = {}

        for variable in lista_variable:
            if not self.variable_pred:
                st.error('Seleccione una variable a predecir')

            if variable != self.variable_pred and 'str' in str(type(self.datos[variable][0])):
                self.mapeo_variable_tipo_string_a_int(variable)

            elif variable != self.variable_pred:
                self.escalar_variable_int(opcion, variable)

    def escalar_variable_int(self, opcion, variable):
        if opcion == 'StandardScaler':
            scaler = StandardScaler()

        elif opcion == 'MinMaxScaler':
            scaler = MinMaxScaler()

        elif opcion == 'RobustScaler':
            scaler = RobustScaler()

        self.x_train[variable] = scaler.fit_transform(pd.DataFrame(self.x_train[variable]))
        self.diccionario_variables[variable] = 'float'
        self.diccionario_scaler[variable] = scaler

    def radio_seleccion_tipo_clasificador(self):
        opcion = st.radio('Seleccione el algoritmo para realizar la predicción',
                          ['None', 'Árbol de Decisión', 'SVC', 'KNN'],
                          key='selectbox_clasificador')
        if opcion == 'None':
            st.error('Debe seleccionar un clasificador')
        else:
            if opcion == 'Árbol de Decisión':
                self.decision_tree_cl()
            elif opcion == 'SVC':
                self.svc()
            elif opcion == 'KNN':
                self.knn()

    def radio_seleccion_tipo_regresion(self):
        opcion = st.radio('Seleccione el algoritmo para realizar la predicción',
                          ['None', 'Regresión Lineal', 'Regresión Logística'],
                          key='selectbox_regresor')
        if opcion == 'None':
            st.error('Debe seleccionar un regresor')

        else:
            self.mostrar_inputs()
            if opcion == 'Regresión Lineal':
                self.linear_regression()

            elif opcion == 'Regresión Logística':
                self.logistic_regression()
            # elif opcion == 'KNN':
            #     self.knn()

    def mostrar_inputs(self):
        self.valores_escalados = {}
        st.warning("Inserte los datos para realizar la predicción")

        for k, v in self.diccionario_variables.items():
            if v == 'float':
                valor = st.number_input(f"Inserte {k}: ")
                valor_escalado = self.diccionario_scaler[k].transform(pd.DataFrame([valor]))[0][0]
                self.valores_escalados[k] = valor_escalado

            else:
                opcion = st.selectbox(f'Escoja uno de los posibles valores de la variable {k}',
                                      list(self.diccionario_variables[k].keys()))

                self.valores_escalados[k] = self.diccionario_variables[k][opcion]

    def decision_tree_cl(self):
        self.classifier = 'tree_cl'
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.Y_train)

        self.get_accuracy(dt)
        self.mostrar_inputs()
        self.obtener_resultado(dt)

    def percentage_accuracy(self, acc):
        return f'{str(np.round(acc, 2)*100)}%'

    def get_accuracy(self, clf):
        if self.classifier == 'svc':
            y_pred = clf.predict(self.X_test)
            y_test = self.Y_test
        else:
            y_pred = [np.argmax(x) for x in clf.predict(self.X_test)]
            y_test = [np.argmax(x) for x in self.Y_test]

        accuracy = balanced_accuracy_score(y_test, y_pred)
        if accuracy > 0.75:
            st.success(f'Se ha conseguido una accuracy del {self.percentage_accuracy(accuracy)}')

        elif 0.6 <= accuracy <= 0.75:
            st.warning(f'Se ha conseguido una accuracy del {self.percentage_accuracy(accuracy)}')

        elif accuracy < 0.6:
            st.error(f'Se ha conseguido una accuracy del {self.percentage_accuracy(accuracy)}')

    def svc(self):
        self.classifier = 'svc'
        svc = SVC()
        self.Y_train = [self.key_de_value(self.mapping, np.argmax(x)) for x in self.Y_train]
        self.Y_test = [self.key_de_value(self.mapping, np.argmax(x)) for x in self.Y_test]

        svc.fit(self.X_train, self.Y_train)
        self.get_accuracy(svc)
        self.obtener_resultado(svc)

    def knn(self):
        self.classifier = 'knn'
        n_vecinos = None
        n_vecinos = int(st.number_input('Introduzca el número de vecinos (número entero): ', value=self.clases))
        if n_vecinos:
            knn = KNeighborsClassifier(n_neighbors=n_vecinos)
            knn.fit(self.X_train, self.Y_train)

            self.get_accuracy(knn)
            self.mostrar_inputs()
            self.obtener_resultado(knn)

    def linear_regression(self):
        self.classifier = 'linear_reg'
        lin_reg = LinearRegression()

        lin_reg.fit(self.X_train, self.Y_train)
        self.obtener_resultado(lin_reg)

    def logistic_regression(self):
        self.classifier = 'logistic_reg'
        tol = float(st.number_input('Introduzca la tolerancia: '))
        solver = st.selectbox('Seleccione el tipo de solver: ',
                               ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])

        penalty = st.selectbox('Seleccione el tipo de penalty: ',
                               [None, 'l1', 'l2', 'elasticnet'])
        if penalty:
            log_reg = LogisticRegression(penalty, tol=tol, solver=solver)
            log_reg.fit(self.X_train, self.Y_train)
            self.obtener_resultado(log_reg)


    def obtener_resultado(self, clf):
        self.resultado = None
        st.session_state['clicked_predecir_resultado'] = False
        if st.button('Predecir resultado', key='predecir_resultado') or st.session_state['clicked_predecir_resultado']:
            st.session_state['clicked_predecir_resultado'] = True

            prediccion = clf.predict([list(self.valores_escalados.values())])

            if self.classifier == 'tree_cl' or self.classifier == 'knn':
                self.resultado = self.key_de_value(self.mapping, np.argmax(prediccion))

            elif self.classifier == 'svc':
                self.resultado = prediccion[0]

            elif self.classifier in ['linear_reg', 'logistic_reg']:
                self.resultado = abs(np.round(prediccion[0]))

            st.success(f"El resultado de la predicción introducida es: {self.resultado}")
            st.balloons()

    def key_de_value(self, diccionario, valor):
        keys = [k for k, v in diccionario.items() if v == valor]
        if keys:
            return keys[0]
        return None

if __name__ == '__main__':
    data = DataTFM()
    data.tfm()