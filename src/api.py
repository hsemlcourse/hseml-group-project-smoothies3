import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Инициализация FastAPI
app = FastAPI(
    title='Jordan Sneaker Resale Profitability API',
    description='FastAPI endpoint for predicting if an Air Jordan sneaker transaction will be profitable.',
    version='1.0',
)

# 2. Загрузка сохранённой champion-модели и энкодеров
MODEL_PATH = 'models/model.pkl'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f'Saved model not found at {MODEL_PATH}. Please run src/train_and_save.py first.')

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
label_encoders = model_data['label_encoders']
feature_cols = model_data['feature_cols']
categorical_cols = model_data['categorical_cols']


# 3. Описание схемы входящего запроса (Pydantic)
class TransactionRequest(BaseModel):
    shoe_model: str = Field(..., example='Jordan 4 Retro')
    colorway: str = Field(..., example='Bred')
    condition: str = Field(..., example='Deadstock')
    sales_channel: str = Field(..., example='GOAT')
    size: str = Field(..., example='9.5')
    retail_price_usd: float = Field(..., example=210.0)
    days_in_inventory: int = Field(..., example=15)
    sale_date: str = Field(..., example='2024-08-15')


# 4. Функция робастного кодирования признаков
def encode_value(value: str, encoder) -> int:
    val_str = str(value).strip().lower()
    classes_lower = [str(c).strip().lower() for c in encoder.classes_]
    if val_str in classes_lower:
        return classes_lower.index(val_str)
    # Если категория неизвестная - ищем наиболее близкое совпадение или берем нулевой класс
    for i, c in enumerate(classes_lower):
        if val_str in c or c in val_str:
            return i
    return 0


# 5. Эндпоинты
@app.get('/')
def read_root():
    return {'status': 'online', 'model': 'Extra Trees Classifier (Champion)', 'features_required': feature_cols}


@app.post('/predict')
def predict_profitability(req: TransactionRequest):
    try:
        # 1. Приведение входящих данных к DataFrame для единообразия
        raw_data = {
            'shoe_model': req.shoe_model,
            'colorway': req.colorway,
            'condition': req.condition,
            'sales_channel': req.sales_channel,
            'size': req.size,
            'retail_price_usd': req.retail_price_usd,
            'days_in_inventory': req.days_in_inventory,
            'sale_date': req.sale_date,
        }

        df = pd.DataFrame([raw_data])

        # 2. Воспроизводим Feature Engineering
        # Quarter
        try:
            sale_date_dt = pd.to_datetime(df['sale_date'], errors='coerce')
            df['quarter'] = sale_date_dt.dt.quarter.fillna(1).astype(int)
        except Exception:
            df['quarter'] = 1

        # Size features
        df['size_num'] = pd.to_numeric(df['size'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(
            9.0
        )
        df['is_common_size'] = df['size_num'].between(8, 11).astype(int)

        def categorize_size(s):
            if pd.isna(s):
                return 'Men'
            if s < 4:
                return 'Toddler_GS'
            if s < 7:
                return 'Women'
            return 'Men'

        df['size_category'] = df['size_num'].apply(categorize_size)

        # 3. Выборка и кодирование признаков
        X_infer = df[feature_cols].copy()

        for col in categorical_cols:
            le = label_encoders[col]
            X_infer[col] = X_infer[col].apply(lambda x: encode_value(x, le))

        # 4. Прогноз
        prediction = int(model.predict(X_infer)[0])
        probability = float(model.predict_proba(X_infer)[0][1])

        result_label = 'Прибыльная' if prediction == 1 else 'Убыточная'

        return {
            'prediction': prediction,
            'result': result_label,
            'probability_profitable': probability,
            'probability_unprofitable': 1.0 - probability,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Inference error: {str(e)}')
