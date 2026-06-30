import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from catboost import Pool

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Прогноз при раке лёгкого", layout="wide")
st.title(" Прогноз выживаемости при раке лёгкого")
st.markdown("---")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
st.sidebar.header("📋 Введите данные пациента")

# 1. ПОЛ: 0 - МУЖСКОЙ, 1 - ЖЕНСКИЙ
gender = st.sidebar.selectbox(
    "Пол",
    options=[0, 1],
    format_func=lambda x: "Мужской" if x == 0 else "Женский"
)

# 2. СТАДИЯ: [0.0, 1.0, 2.0, 3.0, 4.0]
stage = st.sidebar.selectbox(
    "Стадия рака",
    options=[0.0, 1.0, 2.0, 3.0, 4.0],
    format_func=lambda x: f"Стадия {int(x)}" if x > 0 else "Не указана"
)

# 3. ОБЛАСТЬ: 16 значений
region_options = ['0', '01', '02', '1', '2', '21', '3', '32', '4', '5', '6', '7', '8', '81', '82', '9']
region = st.sidebar.selectbox(
    "Область поражения (код)",
    options=region_options
)

# 4. ГИСТОЛОГИЯ: 18 значений
histology_options = [
    '8000', '8001', '8010', '8012', '8021', '8040', '8041', 
    '8070', '8071', '8072', '8074', '8075', '8140', '8240', 
    '8250', '8560', 'Other', 'Unknown'
]
histology = st.sidebar.selectbox(
    "Гистологический тип (код)",
    options=histology_options
)

# 5. ГОД УСТАНОВЛЕНИЯ ДИАГНОЗА: 2001-2022
diagnosis_year = st.sidebar.number_input(
    "Год установления диагноза",
    min_value=2001,
    max_value=2022,
    value=2010,
    step=1
)

# 6. МЕСЯЦ УСТАНОВЛЕНИЯ ДИАГНОЗА: 1-12
diagnosis_month = st.sidebar.selectbox(
    "Месяц установления диагноза",
    options=list(range(1, 13)),
    format_func=lambda x: f"{x} месяц"
)

# 7. ВОЗРАСТ ДИАГНОЗА: 17-98 лет
age = st.sidebar.slider(
    "Возраст на момент диагноза (лет)",
    min_value=17.0,
    max_value=98.0,
    value=65.0,
    step=0.1
)

# 8. ИЗВЕСТНА СТАДИЯ: 0 или 1
known_stage = st.sidebar.selectbox(
    "Стадия известна?",
    options=[0, 1],
    format_func=lambda x: "Да" if x == 1 else "Нет"
)

# --- ПРЕОБРАЗОВАНИЕ ВВОДА В ФОРМАТ МОДЕЛИ ---
patient_data = pd.DataFrame({
    'ПОЛ': [gender],
    'СТАДИЯ': [stage],
    'ОБЛАСТЬ': [region],
    'ГИСТОЛОГИЯ': [histology],
    'ГОД УСТАНОВЛЕНИЯ ДИАГНОЗА': [diagnosis_year],
    'МЕСЯЦ УСТАНОВЛЕНИЯ ДИАГНОЗА': [diagnosis_month],
    'ВОЗРАСТ ДИАГНОЗА': [age],
    'ИЗВЕСТНА СТАДИЯ': [known_stage]
})

# --- КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ---
cat_features = ['ПОЛ', 'ОБЛАСТЬ', 'ГИСТОЛОГИЯ', 'МЕСЯЦ УСТАНОВЛЕНИЯ ДИАГНОЗА', 'ИЗВЕСТНА СТАДИЯ']

for col in cat_features:
    patient_data[col] = patient_data[col].astype(str)

patient_pool = Pool(data=patient_data, cat_features=cat_features)

# --- ОСНОВНАЯ ОБЛАСТЬ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Прогноз для пациента")
    
    if st.button("🔮 Рассчитать прогноз", type="primary"):
        risk = model.predict(patient_pool)[0]
        
        st.metric("Относительный риск (Hazard Ratio)", f"{risk:.3f}")
        
        if risk < 0.5:
            st.success("✅ Низкий риск")
        elif risk < 1.0:
            st.info("📊 Средний риск")
        else:
            st.error("⚠️ Высокий риск")
        
        # Кривая выживаемости (заглушка)
        times = np.linspace(0, 60, 100)
        survival = np.exp(-risk * times / 30)
        
        fig = px.line(
            x=times, y=survival,
            labels={'x': 'Время после диагноза (месяцы)', 'y': 'Вероятность выжить'},
            title="Прогнозируемая кривая выживаемости"
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Информация о пациенте")
    st.write("**Введённые данные:**")
    
    display_dict = {
        "Пол": "Мужской" if gender == 0 else "Женский",
        "Стадия": f"Стадия {int(stage)}" if stage > 0 else "Не указана",
        "Область (код)": region,
        "Гистология (код)": histology,
        "Год диагноза": diagnosis_year,
        "Месяц диагноза": diagnosis_month,
        "Возраст": f"{age:.1f} лет",
        "Стадия известна": "Да" if known_stage == 1 else "Нет"
    }
    
    for key, value in display_dict.items():
        st.write(f"**{key}:** {value}")
    
    st.divider()
    st.caption("Модель: CatBoost (Cox)")

# --- ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ---
st.divider()
with st.expander("ℹ️ О модели и признаках"):
    st.write("""
    **Модель:** CatBoostRegressor с loss_function='Cox'
    
    **Признаки:**
    - **ПОЛ:** 0 — мужской, 1 — женский
    - **СТАДИЯ:** 0.0–4.0
    - **ОБЛАСТЬ:** коды анатомических областей (16 значений)
    - **ГИСТОЛОГИЯ:** коды морфологических типов (18 значений)
    - **ГОД УСТАНОВЛЕНИЯ ДИАГНОЗА:** 2001–2022
    - **МЕСЯЦ УСТАНОВЛЕНИЯ ДИАГНОЗА:** 1–12
    - **ВОЗРАСТ ДИАГНОЗА:** 17–98 лет
    - **ИЗВЕСТНА СТАДИЯ:** 0 — неизвестна, 1 — известна
    
    **Категориальные признаки:** ПОЛ, ОБЛАСТЬ, ГИСТОЛОГИЯ, МЕСЯЦ УСТАНОВЛЕНИЯ ДИАГНОЗА, ИЗВЕСТНА СТАДИЯ
    """)
