import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from lifelines import KaplanMeierFitter

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Прогноз при раке лёгкого", layout="wide")
st.title("🧬 Прогноз выживаемости при раке лёгкого")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    df = pd.read_csv('data_preprocessed.csv')  # путь к вашим данным
    return df

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Загружаем данные и модель
df = load_data()
model = load_model()

# --- БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
st.sidebar.header("📋 Введите данные пациента")

age = st.sidebar.slider("Возраст на момент диагноза", 20, 90, 65)
gender = st.sidebar.selectbox("Пол", ["Мужской", "Женский"])
stage = st.sidebar.selectbox("Стадия рака", ["I", "II", "III", "IV"])
histology = st.sidebar.selectbox("Гистологический тип", 
                                 ["8070", "8140", "8041", "8010", "8240"])

# --- ОСНОВНАЯ ОБЛАСТЬ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Индивидуальная кривая выживаемости")
    
    # --- ТУТ ВСТАВЛЯЕТЕ ВАШУ ЛОГИКУ ПРОГНОЗА ---
    # Это пример, замените на вашу модель!
    risk_score = age * 0.01 + (stage == "IV") * 0.5 + (stage == "III") * 0.3
    
    # Строим кривую (замените на реальные данные)
    times = np.linspace(0, 60, 100)
    survival = np.exp(-risk_score * times / 30)
    
    fig = px.line(
        x=times, y=survival,
        labels={'x': 'Время после диагноза (месяцы)', 'y': 'Вероятность выжить'},
        title=f"Кривая выживаемости для пациента {age} лет"
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Ключевые показатели")
    st.metric("Прогнозируемая медиана", "~24 мес.")
    st.metric("Риск в 1-й год", "35%")
    st.metric("Всего пациентов в базе", len(df))

# --- ГРУППОВОЙ АНАЛИЗ ---
st.divider()
st.subheader("📊 Анализ популяции")

# График распределения по стадиям
stage_counts = df['СТАДИЯ'].value_counts()
fig2 = px.pie(values=stage_counts.values, names=stage_counts.index, title="Распределение по стадиям")
st.plotly_chart(fig2, use_container_width=True)
