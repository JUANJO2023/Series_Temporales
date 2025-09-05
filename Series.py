# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 20:14:01 2025

@author: 
"""

import streamlit as st 
import math
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import requests
import neurokit2 as nk

##----------------------------------------------------------------
## Objetivo 01
##----------------------------------------------------------------

st.set_page_config(layout="wide")

# Título de la aplicación en Streamlit
st.markdown(
    """
    <style>
    /* Ocultar cabecera, menú y footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fondo general */
    .stApp {
        background-color: #0d1b2a; /* azul petróleo oscuro */
        color: #e0e1dd; /* texto gris claro */
    }

    /* Quitar espacio superior */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1b263b; /* azul grisáceo */
        color: white;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #00b4d8; /* celeste brillante */
        text-align: center;
    }

    /* Botones */
    div.stButton > button {
        background-color: #00b4d8;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #48cae4;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='color: white; text-align: center;'>Visualizador de ECG</h1>",
    unsafe_allow_html=True
)
#Ubicación de logotipo
Logo_url='https://raw.githubusercontent.com/JUANJO2023/Series_Temporales/refs/heads/main/LogoECG.png'

st.sidebar.image(Logo_url)


# Función para graficar con papel ECG calibrado
def plot_ecg_paper(signal, fs, seconds=10, title="ECG"):
    """
    Dibuja un ECG con cuadrícula tipo papel electrocardiográfico.
    Escala: 25 mm/s, 10 mm/mV
    """
    samples = int(seconds * fs)
    ecg = signal[:samples]
    t = np.arange(ecg.shape[0]) / fs

    fig, ax = plt.subplots(figsize=(24,8))
    ax.set_facecolor('white')

    # Escala en tiempo y voltaje
    # Horizontal: 1 mm = 0.04 s
    ax.set_xticks(np.arange(0, seconds, 0.04), minor=True)
    ax.set_xticks(np.arange(0, seconds, 0.20))

    # Vertical: 1 mm = 0.1 mV
    ax.set_ylim(-2, 2)
    ax.set_yticks(np.arange(-2, 2, 0.1), minor=True)
    ax.set_yticks(np.arange(-2, 2, 0.5))

    # Dibujar cuadrícula
    ax.grid(which='minor', color='lightcoral', linewidth=0.5, alpha=0.6)
    ax.grid(which='major', color='red', linewidth=1)

    # Relación de aspecto 1:1 (para que 1 mm en X = 1 mm en Y)
    ax.set_aspect(0.04/0.1)

    # Señal ECG
    ax.plot(t, ecg, color='black', linewidth=1)

    # Etiquetas
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Tiempo (s) (Velocidad: 25 mm/s)")
    ax.set_ylabel("Amplitud (mV) (10 mm/mV)")

    return fig

# Función para graficar las 12 derivaciones en orden estándar en una sola cuadrícula
def plot_ecg_12leads_standard(signals, fs, sig_names, seconds=10, record_name="ECG"):
    """
    Dibuja las 12 derivaciones en orden clínico estándar en una sola cuadrícula,
    apiladas verticalmente como en un ECG impreso.
    """
    samples = int(seconds * fs)
    t = np.arange(samples) / fs

    # Orden clínico estándar
    lead_order = ["V6", "V5", "V4", "V3", "V2", "V1",
                  "aVF", "aVL", "aVR", "III", "II", "I"]

    # Filtrar solo las derivaciones disponibles y reordenarlas
    available_leads = [lead for lead in lead_order if lead in sig_names]
    indices = [sig_names.index(lead) for lead in available_leads]

    # Ajustamos tamaño de la figura y separación entre señales
    fig, ax = plt.subplots(figsize=(48, 16))  
    ax.set_facecolor("white")

    # Configuración de cuadrícula ECG
    ax.set_xticks(np.arange(0, seconds, 0.04), minor=True)
    ax.set_xticks(np.arange(0, seconds, 0.20))
    ax.set_yticks(np.arange(-5, 20, 0.1), minor=True)
    ax.set_yticks(np.arange(-5, 20, 0.5))

    ax.grid(which="minor", color="lightcoral", linewidth=0.5, alpha=0.6)
    ax.grid(which="major", color="red", linewidth=1)

    ax.set_aspect(0.04 / 0.1)

    # Graficar cada derivación con poca separación
    offset_step = 1.0  # ahora mucho más compacto
    for i, idx in enumerate(indices):
        ecg = signals[:samples, idx] + i * offset_step
        ax.plot(t, ecg, color="black", linewidth=1)
        # Etiqueta de la derivada
        ax.text(-0.3, i * offset_step, sig_names[idx], va="center", ha="right",
                fontsize=10, color="blue")

    # Etiquetas generales
    ax.set_xlim([0, seconds])
    ax.set_ylim([-1, len(indices) * offset_step])
    ax.set_xlabel("Tiempo (s) (25 mm/s)")
    ax.set_ylabel("Amplitud (10 mm/mV + offset)")
    ax.set_title(f"ECG de {record_name} - 12 derivaciones (orden estándar)", fontsize=16)

    return fig

#Leer registro de directorios y diagnósticos
dfListado = pd.read_csv("https://raw.githubusercontent.com/JUANJO2023/Series_Temporales/refs/heads/main/Directorio-ecg.csv", delimiter=",")
dfDx=pd.read_csv("https://physionet.org/files/ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv")

# Columna 1 = nombre del archivo (ej. JS00001)
opciones = dfListado.iloc[:,1].tolist()

# Inicializar num_registro si no existe
if "num_registro" not in st.session_state:
    st.session_state.num_registro = 0   # posición inicial (puede ser 0 o 240)

# Selectbox de registros
seleccion = st.sidebar.selectbox(
    "Historia clínica",
    opciones,
    index=st.session_state.num_registro
)

# Actualizar session_state según búsqueda
if seleccion:
    st.session_state.num_registro = opciones.index(seleccion)

# Slider que se mantiene sincronizado
#num=st.slider("Número de registro",1,len(dfListado),value=st.session_state.num_registro+1,step=1)-1

# Descargar archivos y leer con wfdb
record_name = dfListado.iloc[st.session_state.num_registro, 1]   # Nombre base (ej. JS00001)
url_base = dfListado.iloc[st.session_state.num_registro, 2]      # URL sin extensión

# Descargar ambos archivos
for ext in [".hea", ".mat"]:
    url = url_base + ext
    r = requests.get(url)
    with open(record_name + ext, "wb") as f:
        f.write(r.content)

# Leer registro descargado
signals, fields = wfdb.rdsamp(record_name)

fs = fields['fs']
sig_name = fields['sig_name']
comentarios=fields['comments']

scomentarios = ""
for i in range(len(comentarios)):
    if i == 2:  # la posición Dx
        codigos = comentarios[i].replace("Dx: ", "").split(",")
        scomentarios += "Dx:\n"
        for codigo in codigos:
            codigo = codigo.strip()
            scomentarios += f"{codigo} - "
            resultado = dfDx.loc[dfDx['Snomed_CT'] == int(codigo), 'Full Name']
            if not resultado.empty:
                scomentarios += f"{resultado.values[0]}\n"
            else:
                scomentarios += "-\n"
        scomentarios += "\n"    
    else:
        scomentarios += comentarios[i] + "\n"

st.sidebar.text(scomentarios)


# Selectbox de tipo de gráfica
sel_grafica = st.sidebar.selectbox(
    "Tipo de Gráfica",
    ["12-lead en una gráfica","Una gráfica por señal"],
    index=0
)

if sel_grafica=="12-lead en una gráfica":
    # Graficar las 12 derivaciones en una sola figura
    fig_12 = plot_ecg_12leads_standard(signals, fs, sig_name, seconds=10, record_name=record_name)
    st.pyplot(fig_12)
else:
    # Graficar cada canal en papel ECG
    for i in range(signals.shape[1]):
        fig = plot_ecg_paper(signals[:, i], fs, seconds=10, title=f"Paciente: {record_name} - {sig_name[i]}")
        st.pyplot(fig)

##----------------------------------------------------------------
## Objetivo 02
##----------------------------------------------------------------

# Seleccionar la derivada apropiada (Recomendado Lead II si existe)
canal = None
for i, name in enumerate(sig_name):
    if "II" in name.upper():  # buscar derivada II
        canal = i
        break

if canal is None:
    canal = 0  # fallback: usar primer canal si no existe Lead II

# Señal seleccionada
ecg_signal = signals[:, canal]

# Procesamiento con neurokit2
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
signals_nk, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)

# Calcular frecuencia cardíaca promedio
hr = signals_nk["ECG_Rate"].mean()

# Mostrar frecuencia cardíaca
st.subheader("📊 Análisis de frecuencia cardíaca")
st.metric(f"Frecuencia cardíaca (promedio) - Derivada {sig_name[canal]}", f"{hr:.1f} lpm ")

# Alerta si fuera del rango normal
if hr < 60 or hr > 100:
    st.error("⚠️ Frecuencia cardíaca fuera del rango normal (60-100 lpm)")
else:
    st.success("✅ Frecuencia cardíaca dentro del rango normal (60-100 lpm)")

# Graficar ECG con picos R detectados
fig_hr = nk.events_plot(info["ECG_R_Peaks"], ecg_cleaned)
st.pyplot(fig_hr)
