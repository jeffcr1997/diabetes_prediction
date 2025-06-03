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
    st.markdown("Carga tu modelo PMML e ingresa datos para realizar predicciones")
    
    # Sidebar para cargar el archivo
    with st.sidebar:
        st.header("📁 Cargar Modelo")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo PMML",
            type=['pmml'],
            help="Sube un archivo con extensión .pmml"
        )
        
        if uploaded_file is not None:
            st.success(f"Archivo cargado: {uploaded_file.name}")
    
    # Inicializar session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    
    # Cargar modelo si se ha subido un archivo
    if uploaded_file is not None and st.session_state.model is None:
        try:
            # Leer el contenido del archivo
            bytes_data = uploaded_file.getvalue()
            
            # Crear modelo PMML
            model = Model.fromString(bytes_data.decode('utf-8'))
            st.session_state.model = model
            
            # Obtener información del modelo
            try:
                # Intentar obtener información básica del modelo
                input_fields = model.inputFields
                target_fields = model.outputFields
                
                st.session_state.model_info = {
                    'input_fields': input_fields,
                    'target_fields': target_fields,
                    'model_name': uploaded_file.name
                }
                
                st.sidebar.success("✅ Modelo cargado exitosamente")
                
            except Exception as e:
                st.sidebar.warning(f"Modelo cargado, pero no se pudo extraer información completa: {str(e)}")
                st.session_state.model_info = {
                    'input_fields': [],
                    'target_fields': [],
                    'model_name': uploaded_file.name
                }
                
        except Exception as e:
            st.sidebar.error(f"Error al cargar el modelo: {str(e)}")
            st.session_state.model = None
            st.session_state.model_info = None
    
    # Mostrar interfaz principal
    if st.session_state.model is not None:
        st.success("🎯 Modelo cargado y listo para predicciones")
        
        # Crear tabs
        tab1, tab2, tab3 = st.tabs(["📊 Predicción Individual", "📋 Predicción por Lotes", "ℹ️ Información del Modelo"])
        
        with tab1:
            st.header("Ingresa los datos para predicción")
            
            # Formulario dinámico para entrada de datos
            with st.form("prediction_form"):
                st.subheader("Parámetros de entrada")
                
                # Si tenemos información de campos, crear inputs dinámicos
                input_data = {}
                
                if st.session_state.model_info['input_fields']:
                    for field in st.session_state.model_info['input_fields']:
                        field_name = field.name if hasattr(field, 'name') else str(field)
                        
                        # Determinar tipo de input basado en el tipo de campo
                        try:
                            field_type = field.dataType if hasattr(field, 'dataType') else 'double'
                            
                            if field_type in ['integer', 'int']:
                                input_data[field_name] = st.number_input(
                                    f"{field_name}",
                                    value=0,
                                    step=1,
                                    format="%d"
                                )
                            elif field_type in ['double', 'float']:
                                input_data[field_name] = st.number_input(
                                    f"{field_name}",
                                    value=0.0,
                                    format="%.6f"
                                )
                            elif field_type == 'string':
                                input_data[field_name] = st.text_input(f"{field_name}")
                            else:
                                input_data[field_name] = st.number_input(
                                    f"{field_name}",
                                    value=0.0,
                                    format="%.6f"
                                )
                        except:
                            # Fallback a número flotante
                            input_data[field_name] = st.number_input(
                                f"{field_name}",
                                value=0.0,
                                format="%.6f"
                            )
                else:
                    # Si no tenemos información de campos, permitir entrada manual
                    st.info("No se pudo extraer información de campos automáticamente. Ingresa los datos manualmente:")
                    
                    num_fields = st.number_input("Número de características", min_value=1, max_value=50, value=5)
                    
                    for i in range(int(num_fields)):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            field_name = st.text_input(f"Nombre campo {i+1}", value=f"feature_{i+1}")
                        with col2:
                            field_value = st.number_input(f"Valor {i+1}", value=0.0, format="%.6f")
                        
                        if field_name:
                            input_data[field_name] = field_value
                
                submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary")
                
                if submitted and input_data:
                    try:
                        # Crear DataFrame con los datos de entrada
                        df_input = pd.DataFrame([input_data])
                        
                        # Realizar predicción
                        prediction = st.session_state.model.predict(df_input)
                        
                        # Mostrar resultados
                        st.success("✅ Predicción realizada exitosamente")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📥 Datos de entrada:")
                            st.json(input_data)
                        
                        with col2:
                            st.subheader("📤 Resultado de la predicción:")
                            if isinstance(prediction, pd.DataFrame):
                                st.dataframe(prediction)
                            elif isinstance(prediction, np.ndarray):
                                st.write(prediction.tolist())
                            else:
                                st.write(prediction)
                        
                    except Exception as e:
                        st.error(f"Error en la predicción: {str(e)}")
                        st.code(traceback.format_exc())
        
        with tab2:
            st.header("Predicción por lotes")
            st.markdown("Carga un archivo CSV con múltiples registros para predicción masiva")
            
            uploaded_csv = st.file_uploader("Selecciona archivo CSV", type=['csv'])
            
            if uploaded_csv is not None:
                try:
                    df_batch = pd.read_csv(uploaded_csv)
                    
                    st.subheader("Vista previa de los datos:")
                    st.dataframe(df_batch.head())
                    
                    if st.button("🚀 Ejecutar predicciones en lote", type="primary"):
                        try:
                            predictions = st.session_state.model.predict(df_batch)
                            
                            st.success(f"✅ {len(df_batch)} predicciones realizadas exitosamente")
                            
                            # Combinar datos originales con predicciones
                            if isinstance(predictions, pd.DataFrame):
                                result_df = pd.concat([df_batch, predictions], axis=1)
                            else:
                                result_df = df_batch.copy()
                                result_df['prediction'] = predictions
                            
                            st.subheader("📊 Resultados:")
                            st.dataframe(result_df)
                            
                            # Botón para descargar resultados
                            csv_buffer = io.StringIO()
                            result_df.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="📥 Descargar resultados (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name="predicciones_resultado.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error en predicción por lotes: {str(e)}")
                            st.code(traceback.format_exc())
                            
                except Exception as e:
                    st.error(f"Error al leer el archivo CSV: {str(e)}")
        
        with tab3:
            st.header("Información del modelo")
            
            if st.session_state.model_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Detalles generales")
                    st.write(f"**Nombre del archivo:** {st.session_state.model_info['model_name']}")
                    st.write(f"**Campos de entrada:** {len(st.session_state.model_info['input_fields'])}")
                    st.write(f"**Campos de salida:** {len(st.session_state.model_info['target_fields'])}")
                
                with col2:
                    st.subheader("🔍 Campos de entrada")
                    if st.session_state.model_info['input_fields']:
                        for field in st.session_state.model_info['input_fields']:
                            field_name = field.name if hasattr(field, 'name') else str(field)
                            field_type = field.dataType if hasattr(field, 'dataType') else 'N/A'
                            st.write(f"• **{field_name}** ({field_type})")
                    else:
                        st.info("Información de campos no disponible")
                
                st.subheader("🎯 Campos de salida")
                if st.session_state.model_info['target_fields']:
                    for field in st.session_state.model_info['target_fields']:
                        field_name = field.name if hasattr(field, 'name') else str(field)
                        st.write(f"• **{field_name}**")
                else:
                    st.info("Información de campos de salida no disponible")
    
    else:
        # Instrucciones cuando no hay modelo cargado
        st.info("👈 Por favor, carga un archivo PMML desde la barra lateral para comenzar")
        
        with st.expander("ℹ️ ¿Qué es PMML?"):
            st.markdown("""
            **PMML (Predictive Model Markup Language)** es un estándar XML para representar modelos de minería de datos y machine learning.
            
            **Características:**
            - Formato estándar e interoperable
            - Soporta múltiples algoritmos (regresión, clasificación, clustering, etc.)
            - Permite portabilidad entre diferentes plataformas
            
            **Formatos soportados:**
            - Archivos con extensión `.pmml`
            - Modelos exportados desde R, Python, SAS, SPSS, etc.
            """)
        
        with st.expander("🚀 Instrucciones de uso"):
            st.markdown("""
            1. **Cargar modelo:** Usa la barra lateral para subir tu archivo `.pmml`
            2. **Predicción individual:** Ingresa valores uno por uno en el formulario
            3. **Predicción por lotes:** Carga un archivo CSV con múltiples registros
            4. **Ver información:** Consulta los detalles del modelo cargado
            """)

if __name__ == "__main__":
    main()
