import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo de Bagging pre-entrenado.")

# Crear pestañas para separar la entrada individual del procesamiento por archivos
tab1, tab2 = st.tabs(["Individual", "Carga de Archivo (Excel/CSV)"])

# Cargar recursos comunes
try:
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
except Exception as e:
    st.error(f"Error cargando los modelos u objetos serializados: {e}")

def preprocesar_y_predecir(df_input):
    # 1. Copia de trabajo
    df_procesado = df_input.copy()
    
    # 2. Eliminar variables que no van al modelo (como ID, Año, etc.) si existen
    columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
    df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
    
    # 3. Aplicar One-Hot para Felder
    if 'Felder' in df_procesado.columns:
        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
        else:
            # Fallback
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]
        
        # Eliminar original
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
    else:
        raise ValueError("La columna 'Felder' no se encuentra en los datos.")
        
    # Asegurar que existan todas las columnas que el modelo espera de Felder
    if isinstance(one_hot_transformer, list):
        for col in [c for c in one_hot_transformer if 'Felder_' in c]:
            if col not in df_procesado.columns:
                df_procesado[col] = 0.0
                
    # 4. Normalizar 'Examen_admisión'
    col_admision = 'Examen_admisión' if 'Examen_admisión' in df_procesado.columns else ('Examen_admision' if 'Examen_admision' in df_procesado.columns else None)
    if col_admision:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[[col_admision]])
        df_procesado = df_procesado.drop(columns=[col_admision], errors='ignore')
    else:
        raise ValueError("No se encontró la columna 'Examen_admisión' en los datos.")
        
    # Reordenar columnas
    si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    df_procesado = df_procesado[columnas_ordenadas]
    
    # Realizar predicción
    predicciones = model.predict(df_procesado)
    return predicciones, df_procesado

# --- PESTAÑA 1: ENTRADA INDIVIDUAL ---
with tab1:
    st.header("Datos de Entrada Individual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
    
    if st.button("Realizar Predicción Individual"):
        try:
            df_input = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
            prediccion, df_proc = preprocesar_y_predecir(df_input)
            
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_proc)
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

# --- PESTAÑA 2: CARGA DE ARCHIVO ---
with tab2:
    st.header("Predicción por Lotes (Excel / CSV)")
    st.write("Sube un archivo que contenga al menos las columnas **Felder** y **Examen_admisión**.")
    
    uploaded_file = st.file_uploader("Selecciona un archivo Excel o CSV", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            # Leer el archivo
            if uploaded_file.name.endswith('.csv'):
                df_lote = pd.read_csv(uploaded_file)
            else:
                df_lote = pd.read_excel(uploaded_file)
                
            st.subheader("Vista previa de los datos cargados")
            st.dataframe(df_lote.head())
            
            if st.button("Procesar y Predecir Archivo"):
                # Verificar que existan las columnas necesarias
                col_felder_ok = 'Felder' in df_lote.columns
                col_examen_ok = 'Examen_admisión' in df_lote.columns or 'Examen_admision' in df_lote.columns
                
                if col_felder_ok and col_examen_ok:
                    # Realizar procesamiento y predicciones
                    predicciones, _ = preprocesar_y_predecir(df_lote)
                    
                    # Agregar predicción como nueva columna
                    df_resultado = df_lote.copy()
                    df_resultado['Nota_final_Predicha'] = predicciones
                    
                    st.subheader("Resultados de las Predicciones")
                    st.dataframe(df_resultado)
                    
                    # Permitir descargar los resultados procesados
                    csv = df_resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar resultados en CSV",
                        data=csv,
                        file_name="predicciones_resultados.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("El archivo debe contener las columnas 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
