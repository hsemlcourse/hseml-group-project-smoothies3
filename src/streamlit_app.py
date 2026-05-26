from datetime import datetime

import requests
import streamlit as st

# Настройка страницы
st.set_page_config(page_title='Air Jordan Resale Oracle', layout='centered')

# Кастомные стили для приведения к единому стилю
st.markdown(
    """
    <style>
    /* Единый стиль заголовка */
    h1 {
        color: #ff4b4b;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    /* Цветовые плашки результатов */
    .profit-badge {
        background: #2ecc71;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 15px 0;
    }
    .loss-badge {
        background: #e74c3c;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 15px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ff4b4b;
        margin-top: 5px;
    }
    .metric-label {
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Основной интерфейс
st.markdown('<h1>Air Jordan Resale Oracle</h1>', unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Введите параметры релиза и условий продажи для прогнозирования окупаемости сделки</p>",
    unsafe_allow_html=True,
)

# Форма ввода параметров
col1, col2 = st.columns([2, 1.5])

with col1:
    with st.container(border=True):
        st.subheader('Характеристики кроссовка и условия сделки')

        subcol1, subcol2 = st.columns(2)

        with subcol1:
            shoe_model = st.selectbox(
                'Модель Jordan',
                ['Jordan 1 Retro High OG', 'Jordan 4 Retro', 'Jordan 11 Retro', 'Jordan 3 Retro', 'Jordan 5 Retro'],
            )

            colorway = st.selectbox(
                'Расцветка (Colorway)',
                [
                    'Chicago',
                    'Bred',
                    'Cool Grey',
                    'Travis Scott',
                    'Military Blue',
                    'Fire Red',
                    'University Blue',
                    'Black Cat',
                ],
            )

            condition = st.selectbox('Физическое состояние', ['Deadstock', 'Near Deadstock', 'VNDS', 'Used'])

            sales_channel = st.selectbox('Канал продаж', ['StockX', 'GOAT', 'eBay', 'Flight Club'])

        with subcol2:
            size = st.select_slider(
                'Размер обуви (US)',
                options=['7', '8', '8.5', '9', '9.5', '10', '10.5', '11', '11.5', '12', '13'],
                value='9.5',
            )

            retail_price_usd = st.number_input(
                'Розничная цена ритейла ($)', min_value=50.0, max_value=400.0, value=190.0, step=10.0
            )

            days_in_inventory = st.slider('Дней в наличии на складе', min_value=1, max_value=120, value=30)

            sale_date = st.date_input('Дата планируемой продажи', value=datetime.today())

with col2:
    with st.container(border=True):
        st.subheader('Прогноз искусственного интеллекта')
        st.write('Нажмите кнопку ниже, чтобы запустить скоринг сделки.')

        predict_btn = st.button('Проанализировать сделку', use_container_width=True)

        if predict_btn:
            with st.spinner('Запускаем скоринг-модели...'):
                payload = {
                    'shoe_model': shoe_model,
                    'colorway': colorway,
                    'condition': condition,
                    'sales_channel': sales_channel,
                    'size': str(size),
                    'retail_price_usd': float(retail_price_usd),
                    'days_in_inventory': int(days_in_inventory),
                    'sale_date': sale_date.strftime('%Y-%m-%d'),
                }

                try:
                    response = requests.post('http://localhost:8000/predict', json=payload, timeout=5)

                    if response.status_code == 200:
                        res_data = response.json()
                        prediction = res_data['prediction']
                        prob_profitable = res_data['probability_profitable']

                        st.write('---')

                        if prediction == 1:
                            st.markdown("<div class='profit-badge'>ПРИБЫЛЬНАЯ СДЕЛКА</div>", unsafe_allow_html=True)
                            st.success(
                                'Модель предсказывает высокую окупаемость. Сделка принесёт чистую прибыль с высокой вероятностью.'
                            )
                        else:
                            st.markdown("<div class='loss-badge'>УБЫТОЧНАЯ СДЕЛКА</div>", unsafe_allow_html=True)
                            st.warning(
                                'Модель рекомендует отказаться от сделки. Высокий риск продать кроссовки в убыток относительно ритейла.'
                            )

                        st.write('---')

                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.markdown("<p class='metric-label'>Вероятность прибыли</p>", unsafe_allow_html=True)
                            st.markdown(
                                f"<p class='metric-value'>{prob_profitable * 100:.1f}%</p>", unsafe_allow_html=True
                            )
                        with mcol2:
                            st.markdown("<p class='metric-label'>Вероятность убытка</p>", unsafe_allow_html=True)
                            st.markdown(
                                f"<p class='metric-value' style='color: #8892b0;'>{(1.0 - prob_profitable) * 100:.1f}%</p>",
                                unsafe_allow_html=True,
                            )

                    else:
                        st.error(f'Ошибка API (Код {response.status_code}): {response.text}')

                except requests.exceptions.ConnectionError:
                    st.error(
                        'Не удалось подключиться к FastAPI серверу (http://localhost:8000). Убедитесь, что API запущено и порт 8000 проброшен.'
                    )
                except Exception as e:
                    st.error(f'Неизвестная ошибка: {str(e)}')
