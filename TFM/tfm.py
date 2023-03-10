import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_absolute_percentage_error, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

image_path = "C:/Users/DEEPGAMING/Desktop/DANI/M谩ster Avanzado de Programaci贸n en Python para Hacking, BigData y Machine Learning IV/TFM/c贸digo/imagenes/"


class DataTFM():
    def tfm(self):
        self.home()


    def home(self):
        st.set_page_config(page_icon="馃搳", page_title="TFM de Daniel", layout="wide")
        # st.write(st.session_state)

        # "st session: ",  st.session_state
        st.title(" Interfaz en Streamlit para realizar predicciones")
        self.container = st.container()

        uploaded_file = self.container.file_uploader(
            "_A帽ada el archivo .csv_", type='csv',
            key="uploaded_file",
        )
        if uploaded_file:
            self.data = pd.read_csv(uploaded_file, sep=";")

        else:
            self.container.info(
                f"""馃憜 Debe cargar primero un dato con extensi贸n .csv"""
            )
            st.stop()

        if self.data is not None:
            self.select_prediction()

    def select_prediction(self):
        self.type_prediction = st.radio('Seleccione si desea hacer una clasificaci贸n o una regresi贸n',
                                        ['None', 'Clasificaci贸n', 'Regresi贸n'], key='selectbox_tipo')
        if self.type_prediction != 'None':
            self.predict_variable()
        else:
            st.warning('Seleccione el tipo de predicci贸n a realizar')

    def init_buttons(self):
        self.variables = list(self.data.columns)
        self.cont_nulls = 0

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

        if self.cont_nulls:
            if st.button('Imputar valores', key='imputar_valores') or st.session_state["clicked_imputar_valores"]:
                st.session_state['clicked_imputar_valores'] = True
                self.impute_values()
                st.success('Los datos se han imputado correctamente')

        self.variables_selector = [None] + self.variables

        self.pred_variable = st.selectbox('_Seleccione la variable a predecir_', self.variables_selector)

        if self.pred_variable:
            self.num_unique_values = len(np.unique(self.data[self.pred_variable]))

        if self.type_prediction == 'Clasificaci贸n':
            self.change_variable_pred_int_to_str()

    def describe_dataset(self):
        self.c1, self.c2 = st.columns([3, 3])
        with self.c1:
            st.title("Valores nulos visualmente")
            fig, ax = plt.subplots()
            g = sns.heatmap(self.data.isnull(), cbar=False, ax=ax)
            st.write(fig)

        with self.c2:
            st.title("Estad铆stica descriptiva")
            st.dataframe(self.data.describe(), use_container_width=True)

            dict_data_nulls = dict(self.data.isnull().sum())
            for k, v in dict_data_nulls.items():
                if v > 0:
                    self.cont_nulls += 1
                    st.warning(f'Para la variable {k} hay {v} valores nulos')

    def impute_values(self):
        simple = SimpleImputer(strategy='most_frequent')
        self.data = pd.DataFrame(simple.fit_transform(self.data), columns=self.variables)

    def predict_variable(self):

        self.init_buttons()

        if self.pred_variable is not None:
            if (self.type_prediction == 'Clasificaci贸n' and 'str' in str(type(self.data[self.pred_variable][0]))) or (self.type_prediction == 'Clasificaci贸n' and self.num_unique_values == 2) or (self.type_prediction == 'Regresi贸n' and ('float' in str(type(self.data[self.pred_variable][0]))) or 'int' in 'int' in str(type(self.data[self.pred_variable][0]))):
                st.success('La variable para realizar la predicci贸n seleccionada es: ' + self.pred_variable)
                if st.button('Mostrar datos', key='mostrar_datos') or st.session_state["clicked_mostrar_datos"]:
                    st.session_state["clicked"] = True
                    st.session_state["clicked_mostrar_datos"] = True
                    st.session_state["selected_var_pred"] = True

                    self.list_variables = list(self.data.columns)

                    if self.type_prediction:
                        self.show_data()
                    else:
                        st.warning('Seleccione el tipo de predicci贸n a realizar')

                    if st.button('Ver gr谩ficas', key='ver_graficas') or st.session_state["clicked_ver_graficas"]:
                        st.session_state['clicked_ver_graficas'] = True
                        if self.pred_variable in self.variables:
                            self.variables.remove(self.pred_variable)

                        options = st.multiselect(label='Seleccionar las diferentes variables para realizar las gr谩ficas'
                                                 , options=self.variables)

                        self.graph_combinations = list(combinations(options, 2))

                        st.markdown('Se va a hacer un plot de las diferentes clasificaciones')

                        if self.type_prediction == 'Clasificaci贸n':
                            self.classification_graphs()

                        elif self.type_prediction == 'Regresi贸n':
                            self.regression_graphs()

                    if st.button('Realizar predicciones', key='realizar_predicciones') or st.session_state[
                        'clicked_realizar_predicciones']:
                        st.session_state['clicked_realizar_predicciones'] = True

                        if self.type_prediction == 'Clasificaci贸n':
                            self.select_type_classifier()

                        elif self.type_prediction =='Regresi贸n':
                            self.select_type_regression()

            else:
                st.error('Variable no permitida en esta predicci贸n')

    def classification_graphs(self):
        tab1, tab2, tab3 = st.tabs(["Matriz correlaci贸n", "FacetGrid", "Violinplot"])
        with tab1:
            self.matriz_correlation()

        with tab2:
            self.scatter()

        with tab3:
            self.violinplot()
    
    def regression_graphs(self):
        tab1, tab2, tab3 = st.tabs(["Matriz correlaci贸n", "Boxplot", "Distplot"])
        with tab1:
            self.matriz_correlation()
        with tab2:
            self.boxplot()
        with tab3:
            self.distplot()

    def change_variable_pred_int_to_str(self):
        if self.pred_variable:
            self.unique_values = np.unique(self.data[self.pred_variable])

            if len(self.unique_values) == 2 and str(self.unique_values[0]).isdigit():
                for i in self.unique_values:
                    value = st.text_input(f"Inserte valor para {i}: ")
                    self.data[self.pred_variable] = self.data[self.pred_variable].replace({i: value})

    def distplot(self):
        option_plot = st.selectbox('Seleccione la variable para mostrar el distplot', self.variables)

        if option_plot:
            c1, c2, c3 = st.columns([2, 2, 2])

            with c2:
                fig, ax = plt.subplots()
                if 'str' in str(type(self.data[option_plot][0])):
                    st.error('Variable no num茅rica')
                else:
                    sns.distplot(self.data[option_plot], ax=ax).set(title=f'Displot de {option_plot}')
                    st.write(fig)

    def boxplot(self):
        option_plot = st.selectbox('Seleccione la variable para mostrar el boxplot', self.variables)

        if option_plot:
            c1, c2, c3 = st.columns([2, 2, 2])

            with c2:
                fig, ax = plt.subplots()
                if 'str' in str(type(self.data[option_plot][0])):
                    st.error('Variable no num茅rica')
                else:
                    sns.boxplot(data=self.data, x=option_plot, ax=ax).set(title=f'Boxplot de {option_plot}')
                    st.write(fig)

    def violinplot(self):
        if 'str' in str(type(self.data[self.pred_variable][0])):
            option_plot = st.selectbox('Seleccione la variable para mostrar el violinplot', self.variables)

            if option_plot:
                c1, c2, c3 = st.columns([2, 2, 2])

                with c2:
                    fig, ax = plt.subplots()
                    sns.violinplot(data=self.data, x=option_plot, y=self.pred_variable, inner="stick", ax=ax).set(title=f'Violinplot de {option_plot}')
                    st.write(fig)

        else:
            st.error('La variable seleccionada es num茅rica')

    def matriz_correlation(self):
        try:
            c1, c2, c3 = st.columns([2,2,2])
            with c2:
                fig, ax = plt.subplots()
                sns.heatmap(self.data.corr(), cmap='coolwarm', ax=ax, annot=True, linewidth=.5)
                st.write(fig)
        except Exception as e:
            st.error(f'Error al contruir la matriz de correlaci贸n: {e}')

    def scatter(self):
        if 'str' in str(type(self.data[self.pred_variable][0])):
            selection = st.selectbox('Seleccione la combinaci贸n para mostrar la gr谩fica', self.graph_combinations)
            if selection:
                c1, c2, c3 = st.columns([2, 2, 2])
                with c2:
                    g = sns.FacetGrid(self.data, hue=self.pred_variable, height=3).map(plt.scatter, selection[0],
                                                                                        selection[1]).add_legend()

                    g.fig.suptitle(f'Clasificaci贸n de {self.pred_variable} seg煤n {selection[0]}-{selection[1]}',
                                   fontsize=5)
                    st.pyplot(g)

        else:
            st.error('La variable seleccionada es num茅rica')

    def show_data(self):
        self.num_classes = None
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.session_state["clicked_mostrar_datos"]==True:
                st.markdown('Datos originales')
                st.dataframe(self.data)

                option = st.selectbox('Seleccione c贸mo escalar los datos: ',
                                      [None, 'StandardScaler', 'MinMaxScaler', 'RobustScaler'])

                if option:
                    porcentaje_test = float(st.number_input('Introduzca el procentaje de datos para test', value=0.2))
                    if st.button('Transformar datos', key='transformar_datos') or st.session_state["clicked_transformar_datos"]:
                        st.session_state["clicked_transformar_datos"] = True

                        if option:
                            if self.type_prediction == 'Clasificaci贸n':
                                self.transform_data_classification(self.list_variables, option)
                            elif self.type_prediction == 'Regresi贸n':
                                self.transform_data_regression(self.list_variables, option)

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

        if self.num_classes:
            st.info(f"Hay {self.num_classes} clases en el conjunto de datos")

    def transform_data_classification(self, list_variable, option):
        self.y_train_cat = self.data[self.pred_variable]
        self.x_train = self.data.drop(self.pred_variable, axis=1)
        self.dict_variables = {}
        self.dict_scaler = {}

        for variable in list_variable:
            if not self.pred_variable:
                st.error('Seleccione una variable a predecir')

            if variable == self.pred_variable:
                y = np.unique(self.y_train_cat)
                names = y
                self.num_classes = len(names)

                self.mapping = {key: value for key, value in zip(y, range(len(y)))}
                processed_y = np.array([self.mapping[i] for i in self.data[variable]])
                self.y_train = to_categorical(processed_y, num_classes=self.num_classes)

            elif variable != self.pred_variable and 'str' in str(type(self.data[variable][0])):
                self.transform_string_to_int(variable)

            else:
                self.normalize_variable_int(option, variable)

    def transform_string_to_int(self, var):
        y = np.unique(self.data[var])
        mapping = {key: value for key, value in zip(y, range(len(y)))}
        processed_y = np.array([mapping[i] for i in self.data[var]])

        self.x_train[var] = processed_y
        self.dict_variables[var] = mapping

    def transform_data_regression(self, list_variable, option):
        self.y_train = self.data[self.pred_variable]
        self.x_train = self.data.drop(self.pred_variable, axis=1)
        self.dict_variables = {}
        self.dict_scaler = {}

        for variable in list_variable:
            if not self.pred_variable:
                st.error('Seleccione una variable a predecir')

            if variable != self.pred_variable and 'str' in str(type(self.data[variable][0])):
                self.transform_string_to_int(variable)

            elif variable != self.pred_variable:
                self.normalize_variable_int(option, variable)

    def normalize_variable_int(self, option, variable):
        if option == 'StandardScaler':
            scaler = StandardScaler()

        elif option == 'MinMaxScaler':
            scaler = MinMaxScaler()

        elif option == 'RobustScaler':
            scaler = RobustScaler()

        self.x_train[variable] = scaler.fit_transform(pd.DataFrame(self.x_train[variable]))
        self.dict_variables[variable] = 'float'
        self.dict_scaler[variable] = scaler

    def select_type_classifier(self):
        option = st.radio('Seleccione el algoritmo para realizar la predicci贸n',
                          ['None', '脕rbol de Decisi贸n', 'SVC', 'KNN'],
                          key='selectbox_clasificador')
        if option == 'None':
            st.error('Debe seleccionar un clasificador')
        else:
            if option == '脕rbol de Decisi贸n':
                self.decision_tree_cl()
            elif option == 'SVC':
                self.svc()
            elif option == 'KNN':
                self.knn()

    def select_type_regression(self):
        option = st.radio('Seleccione el algoritmo para realizar la predicci贸n',
                          ['None', 'Regresi贸n Lineal', 'KN Regresor'],
                          key='selectbox_regresor')
        if option == 'None':
            st.error('Debe seleccionar un regresor')

        else:
            if option == 'Regresi贸n Lineal':
                self.linear_regression()

            elif option == 'KN Regresor':
                self.knregressor()

            # elif opcion == 'KNN':
            #     self.knn()

    def show_inputs(self):
        self.norm_values = {}
        st.warning("Inserte los datos para realizar la predicci贸n")

        for k, v in self.dict_variables.items():
            possible_values = np.unique(self.data[k])
            if v == 'float' and len(possible_values) == 2:
                value = st.number_input(f"Inserte {k}. Sus posibles valores son {list(possible_values)}: ")
                norm_value = self.dict_scaler[k].transform(pd.DataFrame([value]))[0][0]
                self.norm_values[k] = norm_value

            elif v == 'float':
                value = st.number_input(f"Inserte {k}: ")
                norm_value = self.dict_scaler[k].transform(pd.DataFrame([value]))[0][0]
                self.norm_values[k] = norm_value

            else:
                option = st.selectbox(f'Escoja uno de los posibles valores de la variable {k}',
                                      list(self.dict_variables[k].keys()))

                self.norm_values[k] = self.dict_variables[k][option]

    def decision_tree_cl(self):
        self.classifier = 'tree_cl'
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.Y_train)

        self.get_accuracy(dt)
        self.show_inputs()
        self.get_result(dt)

    def knn(self):
        self.classifier = 'knn_cl'
        n_neighbors = None
        n_neighbors = int(st.number_input('Introduzca el n煤mero de vecinos (n煤mero entero): ', value=self.num_classes))
        if n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(self.X_train, self.Y_train)

            self.get_accuracy(knn)
            self.show_inputs()
            self.get_result(knn)

    def knregressor(self):
        self.classifier = 'knr'
        n_neighbors = None
        n_neighbors = int(st.number_input('Introduzca el n煤mero de vecinos (n煤mero entero): '))

        if n_neighbors:
            knr = KNeighborsRegressor(n_neighbors=n_neighbors)
            knr.fit(self.X_train, self.Y_train)
            self.error_regression(knr)
            self.show_inputs()
            self.get_result(knr)

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

    def error_regression(self, clf):
        y_pred = clf.predict(self.X_test)
        y_test = self.Y_test
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.info(f'El MAE es: {mae}')
        st.info(f'El MAPE es: {mape}')
        st.info(f'El MSE es: {mse}')

    def svc(self):
        self.classifier = 'svc'
        kernel = st.selectbox('Seleccione el kernel a introducir: ', ['rbf', 'linear', 'poly', 'sigmoid'])
        svc = SVC(kernel=kernel)
        self.Y_train = [self.key_from_value(self.mapping, np.argmax(x)) for x in self.Y_train]
        self.Y_test = [self.key_from_value(self.mapping, np.argmax(x)) for x in self.Y_test]

        svc.fit(self.X_train, self.Y_train)
        self.get_accuracy(svc)
        self.show_inputs()
        self.get_result(svc)

    def linear_regression(self):
        self.classifier = 'linear_reg'
        lin_reg = LinearRegression()

        lin_reg.fit(self.X_train, self.Y_train)
        self.error_regression(lin_reg)
        self.show_inputs()
        self.get_result(lin_reg)

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
            self.get_result(log_reg)


    def get_result(self, clf):
        self.result = None
        st.session_state['clicked_predecir_resultado'] = False
        if st.button('Predecir resultado', key='predecir_resultado') or st.session_state['clicked_predecir_resultado']:
            st.session_state['clicked_predecir_resultado'] = True

            pred = clf.predict([list(self.norm_values.values())])

            if self.classifier == 'tree_cl' or self.classifier == 'knn_cl':
                self.result = self.key_from_value(self.mapping, np.argmax(pred))

            elif self.classifier == 'svc':
                self.result = pred[0]

            elif self.classifier in ['linear_reg', 'knr']:
                if self.num_unique_values == 2:
                    self.result = abs(np.round(pred[0]))
                else:
                    self.result = pred[0]

            st.success(f"El resultado de la predicci贸n introducida es: {self.result}")
            st.balloons()

    def key_from_value(self, diccionario, valor):
        keys = [k for k, v in diccionario.items() if v == valor]
        if keys:
            return keys[0]
        return None

if __name__ == '__main__':
    data = DataTFM()
    data.tfm()