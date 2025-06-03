import streamlit as st
import pandas as pd
import numpy as np
from pypmml import Model
import io
import traceback

def main():
    st.set_page_config(
        page_title="Predictor PMML",
        page_icon="🔮",
        layout="wide"
    )

    st.title("🔮 Predictor con Modelos PMML")
    st.markdown("Carga tu modelo PMML e ingresa datos para realizar predicciones.")

    # Sidebar para cargar el archivo
    with st.sidebar:
        st.header("📁 Cargar Modelo")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo `.pmml`",
            type=['pmml'],
            help="Sube un archivo con extensión .pmml"
        )

        if uploaded_file:
            st.success(f"Archivo cargado: {uploaded_file.name}")

    # Inicializar session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None

    # Cargar modelo
    if uploaded_file and st.session_state.model is None:
        try:
            bytes_data = uploaded_file.getvalue()
            model = Model.fromString(bytes_data.decode('utf-8'))
            st.session_state.model = model

            try:
                input_fields = model.inputFields
                target_fields = model.outputFields

                st.session_state.model_info = {
                    'input_fields': input_fields,
                    'target_fields': target_fields,
                    'model_name': uploaded_file.name
                }

                st.sidebar.success("✅ Modelo cargado exitosamente")

            except Exception as e:
                st.sidebar.warning(f"Modelo cargado, pero sin información completa: {str(e)}")
                st.session_state.model_info = {
                    'input_fields': [],
                    'target_fields': [],
                    'model_name': uploaded_file.name
                }

        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
            st.session_state.model = None
            st.session_state.model_info = None

    # Interfaz principal
    if st.session_state.model:
        st.success("🎯 Modelo listo para realizar predicciones")

        tab1, tab2, tab3 = st.tabs(["📊 Predicción Individual", "📋 Predicción por Lotes", "ℹ️ Información del Modelo"])

        # TAB 1 - Predicción individual
        with tab1:
            st.header("📊 Predicción Individual")
            st.markdown("Completa el formulario con los datos de entrada para realizar una predicción.")

            with st.form("prediction_form"):
                st.subheader("📝 Parámetros de Entrada")
                input_data = {}

                if st.session_state.model_info['input_fields']:
                    for field in st.session_state.model_info['input_fields']:
                        name = field.name if hasattr(field, 'name') else str(field)
                        dtype = field.dataType if hasattr(field, 'dataType') else 'double'

                        if dtype in ['integer', 'int']:
                            input_data[name] = st.number_input(f"{name}", value=0, step=1, format="%d")
                        elif dtype in ['double', 'float']:
                            input_data[name] = st.number_input(f"{name}", value=0.0, format="%.6f")
                        elif dtype == 'string':
                            input_data[name] = st.text_input(f"{name}")
                        else:
                            input_data[name] = st.number_input(f"{name}", value=0.0, format="%.6f")
                else:
                    st.info("No se pudo extraer información del modelo. Ingresa los datos manualmente.")
                    num_fields = st.number_input("Número de características", min_value=1, max_value=50, value=5)

                    for i in range(num_fields):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            name = st.text_input(f"Nombre del campo {i+1}", value=f"feature_{i+1}")
                        with col2:
                            value = st.number_input(f"Valor {i+1}", value=0.0, format="%.6f")
                        if name:
                            input_data[name] = value

                submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary")

                if submitted and input_data:
                    try:
                        df_input = pd.DataFrame([input_data])
                        prediction = st.session_state.model.predict(df_input)
                        st.success("✅ Predicción realizada exitosamente")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📥 Datos de entrada")
                            st.json(input_data)

                        with col2:
                            st.subheader("📤 Resultado")
                            if isinstance(prediction, pd.DataFrame):
                                st.dataframe(prediction)
                            elif isinstance(prediction, np.ndarray):
                                st.write(prediction.tolist())
                            else:
                                st.write(prediction)

                    except Exception as e:
                        st.error(f"❌ Error en la predicción: {str(e)}")
                        st.code(traceback.format_exc())

        # TAB 2 - Predicción por lotes
        with tab2:
            st.header("📋 Predicción por Lotes")
            st.markdown("Carga un archivo CSV con múltiples registros para realizar predicciones en lote.")

            uploaded_csv = st.file_uploader("📄 Selecciona archivo CSV", type=['csv'])

            if uploaded_csv:
                try:
                    df_batch = pd.read_csv(uploaded_csv)
                    st.subheader("🔍 Vista previa de los datos")
                    st.dataframe(df_batch.head())

                    if st.button("🚀 Ejecutar predicciones en lote", type="primary"):
                        try:
                            predictions = st.session_state.model.predict(df_batch)
                            st.success(f"✅ {len(df_batch)} predicciones realizadas exitosamente")

                            result_df = pd.concat([df_batch, predictions], axis=1) if isinstance(predictions, pd.DataFrame) else df_batch.assign(prediction=predictions)

                            st.subheader("📊 Resultados")
                            st.dataframe(result_df)

                            buffer = io.StringIO()
                            result_df.to_csv(buffer, index=False)

                            st.download_button(
                                label="📥 Descargar Resultados",
                                data=buffer.getvalue(),
                                file_name="predicciones_resultado.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"❌ Error en la predicción por lotes: {str(e)}")
                            st.code(traceback.format_exc())

                except Exception as e:
                    st.error(f"❌ Error al leer el archivo CSV: {str(e)}")

        # TAB 3 - Información del modelo
        with tab3:
            st.header("ℹ️ Información del Modelo")

            if st.session_state.model_info:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📦 Detalles Generales")
                    st.write(f"**Archivo:** {st.session_state.model_info['model_name']}")
                    st.write(f"**Campos de Entrada:** {len(st.session_state.model_info['input_fields'])}")
                    st.write(f"**Campos de Salida:** {len(st.session_state.model_info['target_fields'])}")

                with col2:
                    st.subheader("🔍 Campos de Entrada")
                    if st.session_state.model_info['input_fields']:
                        for field in st.session_state.model_info['input_fields']:
                            name = field.name if hasattr(field, 'name') else str(field)
                            dtype = field.dataType if hasattr(field, 'dataType') else 'N/A'
                            st.write(f"• **{name}** ({dtype})")
                    else:
                        st.info("No disponible")

                st.subheader("🎯 Campos de Salida")
                if st.session_state.model_info['target_fields']:
                    for field in st.session_state.model_info['target_fields']:
                        name = field.name if hasattr(field, 'name') else str(field)
                        st.write(f"• **{name}**")
                else:
                    st.info("No disponible")

    else:
        st.info("👈 Carga un archivo PMML desde la barra lateral para comenzar.")
        with st.expander("📌 Instrucciones de Uso"):
            st.markdown("""
            1. **Carga tu modelo:** Desde la barra lateral, selecciona un archivo `.pmml`.
            2. **Predicción individual:** Ingresa valores manualmente para hacer una predicción.
            3. **Predicción por lotes:** Sube un archivo `.csv` con múltiples registros.
            4. **Información del modelo:** Revisa los campos de entrada y salida definidos.
            """)

if __name__ == "__main__":
    main()
