import os

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42


def train_and_save():
    print('=== Training Final Model ===')

    # 1. Загрузка данных
    csv_path = 'data/raw/jordan_market_dataset_2026.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset not found at {csv_path}. Please place it there first.')

    df = pd.read_csv(csv_path)

    # 2. Очистка и приведение колонок к нижнему регистру
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

    # Очистка цен
    price_cols = [c for c in df.columns if any(k in c for k in ['price', 'retail', 'resale'])]
    for col in price_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('[$,]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Даты
    date_cols = [c for c in df.columns if 'date' in c or 'release' in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Удаление NaN
    df = df.dropna(subset=price_cols)
    df = df.drop_duplicates()

    # Таргет
    retail_col = next((c for c in df.columns if 'retail' in c), price_cols[0])
    resale_col = next((c for c in df.columns if 'resale' in c), price_cols[1] if len(price_cols) > 1 else price_cols[0])

    df['profit'] = df[resale_col] - df[retail_col]
    df['is_profitable'] = (df['profit'] > 0).astype(int)

    # 3. Feature Engineering
    sale_date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and 'sale' in c), None)
    if sale_date_col:
        df['quarter'] = df[sale_date_col].dt.quarter
    else:
        df['quarter'] = 1

    size_col = next((c for c in df.columns if 'size' in c), None)
    if size_col:
        df['size_num'] = pd.to_numeric(df[size_col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        df['is_common_size'] = df['size_num'].between(8, 11).astype(int)

        def categorize_size(s):
            if pd.isna(s):
                return 'Unknown'
            if s < 4:
                return 'Toddler_GS'
            if s < 7:
                return 'Women'
            return 'Men'

        df['size_category'] = df['size_num'].apply(categorize_size)
    else:
        df['is_common_size'] = 1
        df['size_category'] = 'Men'

    # Список признаков
    feature_cols = [
        'shoe_model',
        'colorway',
        'condition',
        'sales_channel',
        'size',
        'retail_price_usd',
        'days_in_inventory',
        'quarter',
        'is_common_size',
        'size_category',
    ]

    # Заполняем пропуски
    X = df[feature_cols].copy()
    y = df['is_profitable']

    # 4. Кодирование категориальных признаков
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f'Encoded {col}: {list(le.classes_)}')

    X = X.fillna(X.median(numeric_only=True))

    # 5. Обучение финальной champion-модели Extra Trees (GridSearch params)
    print('Fitting ExtraTreesClassifier final champion model...')
    best_params = {'max_depth': None, 'min_samples_leaf': 3, 'n_estimators': 300}
    model = ExtraTreesClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X, y)

    # 6. Сохранение
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
    }

    model_path = 'models/model.pkl'
    joblib.dump(model_data, model_path)
    print(f'Successfully trained final model and saved to {model_path}!')


if __name__ == '__main__':
    train_and_save()
