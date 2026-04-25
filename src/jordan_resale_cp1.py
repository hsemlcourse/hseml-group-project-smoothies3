# CP1-Предсказание прибыльности перепродажи Air Jordan
# Датасет: https://www.kaggle.com/datasets/abdullahmeo/air-jordan-sneaker-market-and-resale-data2023-2026
# Задача: бинарная классификация (is_profitable = resale_price > retail_price)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, ConfusionMatrixDisplay

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_STATE = 42

FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===
# 1. ЗАГРУЗКА ДАННЫХ
# ===
df = pd.read_csv('data/raw/jordan_market_dataset_2026.csv')

print('=== 1. Описание датасета ===')
print(f'Строк:    {df.shape[0]:,}')
print(f'Столбцов: {df.shape[1]}')
print(df.dtypes)
print(df.head())
print(df.describe(include='all'))

# ===
# 2. ОЧИСТКА ДАННЫХ
# ===
print('\n=== 2. Очистка данных ===')

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print('\n-- Пропуски --')
if missing.sum() == 0:
    print('Пропусков нет')
else:
    print(pd.DataFrame({'count': missing, 'pct%': missing_pct})[missing > 0])

n_dups = df.duplicated().sum()
print(f'\nДубликатов: {n_dups}')
df = df.drop_duplicates()

df.columns = (
    df.columns.str.lower().str.strip()
    .str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
)
print(f'Столбцы: {df.columns.tolist()}')

price_cols = [c for c in df.columns if any(k in c for k in ['price', 'retail', 'resale'])]
for col in price_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace('[$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

date_cols = [c for c in df.columns if 'date' in c or 'release' in c]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

before = len(df)
df = df.dropna(subset=price_cols)
print(f'Удалено строк с NaN в ценах: {before - len(df)}, осталось: {len(df):,}')

print('\n-- Выбросы (IQR x3) --')
for col in price_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    n_out = (~df[col].between(lower, upper)).sum()
    df = df[df[col].between(lower, upper)]
    print(f'  {col}: [{lower:.0f}, {upper:.0f}], удалено: {n_out}')

print(f'Итог после очистки: {len(df):,} строк')

# ===
# 3. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ===
print('\n=== 3. Целевая переменная ===')

retail_col = next((c for c in df.columns if 'retail' in c), price_cols[0])
resale_col  = next((c for c in df.columns if 'resale' in c),
                   price_cols[1] if len(price_cols) > 1 else price_cols[0])

df['profit'] = df[resale_col] - df[retail_col]
df['is_profitable'] = (df['profit'] > 0).astype(int)

vc = df['is_profitable'].value_counts()
print(f'Убыточных: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)')
print(f'Прибыльных: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)')

# ===
# 4. FEATURE ENGINEERING
# ===
print('\n=== 4. Feature Engineering ===')

# --- Quarter: квартал из даты продажи ---
# Помогает учесть сезонные циклы покупательской способности
sale_date_col = next((c for c in df.columns
                      if pd.api.types.is_datetime64_any_dtype(df[c]) and 'sale' in c), None)
if sale_date_col:
    df['quarter'] = df[sale_date_col].dt.quarter
    print(f'quarter создан из {sale_date_col}: {df["quarter"].value_counts().to_dict()}')
else:
    print('Столбец с датой продажи не найден — quarter пропущен')

# --- Size Engineering ---
size_col = next((c for c in df.columns if 'size' in c), None)
if size_col:
    # Переводим размер в числовой
    df['size_num'] = pd.to_numeric(
        df[size_col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # is_common_size: ходовые мужские размеры 8–11 US
    # Вресейл чаще всего продаються как раз эти размеры, с большей маржой
    df['is_common_size'] = df['size_num'].between(8, 11).astype(int)

    # size_category: группировка по категории покупателя
    def categorize_size(s):
        if pd.isna(s):    return 'Unknown'
        if s < 4:         return 'Toddler_GS'   # детские
        if s < 7:         return 'Women'         # женские
        return 'Men'                              # мужские

    df['size_category'] = df['size_num'].apply(categorize_size)

    print(f'is_common_size: {df["is_common_size"].value_counts().to_dict()}')
    print(f'size_category:  {df["size_category"].value_counts().to_dict()}')
else:
    print('Столбец size не найден — size features пропущены')

# ===
# 5. РАБОТА С ФИЧАМИ
# ===
print('\n=== 5. Работа с фичами ===')

all_cols      = df.columns.tolist()
target_cols   = ['profit', 'is_profitable']

leak_cols     = [resale_col] + [c for c in df.columns if 'margin' in c]
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
extra_drop    = ['size_num'] if 'size_num' in df.columns else []

feature_cols  = [c for c in all_cols
                 if c not in target_cols + leak_cols + datetime_cols + extra_drop]

print(f'Всего столбцов:          {len(all_cols)}')
print(f'Исключено (утечка):      {leak_cols}')
print(f'Исключено (datetime):    {datetime_cols}')
print(f'Признаков для модели:    {len(feature_cols)} - {feature_cols}')

X = df[feature_cols].copy()
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(X.median(numeric_only=True))
y = df['is_profitable']

print(f'Матрица признаков: {X.shape}')

# ===
# 6. ВИЗУАЛИЗАЦИИ
# ===
print('\n=== 6. Визуализации ===')

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('EDA - Air Jordan Resale Data', fontsize=14)

vc = df['is_profitable'].value_counts()
axes[0,0].bar(['Убыточная (0)', 'Прибыльная (1)'], vc.values,
               color=['#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.5)
axes[0,0].set_title('Баланс классов')
axes[0,0].set_ylabel('Количество сделок')
for i, v in enumerate(vc.values):
    axes[0,0].text(i, v + len(df)*0.005,
                   f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

df['profit'].clip(df['profit'].quantile(0.02), df['profit'].quantile(0.98)).hist(
    bins=50, ax=axes[0,1], color='steelblue', edgecolor='white')
axes[0,1].axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even (0$)')
axes[0,1].set_title('Распределение прибыли ($)')
axes[0,1].set_xlabel('Прибыль ($)')
axes[0,1].legend()

sample = df.sample(min(2000, len(df)), random_state=RANDOM_STATE)
colors = sample['is_profitable'].map({0: '#e74c3c', 1: '#2ecc71'})
axes[1,0].scatter(sample[retail_col], sample[resale_col], c=colors, alpha=0.35, s=12)
max_p = max(df[retail_col].max(), df[resale_col].max())
axes[1,0].plot([0, max_p], [0, max_p], 'k--', lw=1.5, label='Break-even')
axes[1,0].set_xlabel('Розничная цена ($)')
axes[1,0].set_ylabel('Цена перепродажи ($)')
axes[1,0].set_title('Retail vs Resale Price')
axes[1,0].legend(handles=[
    Patch(color='#2ecc71', label='Прибыльная'),
    Patch(color='#e74c3c', label='Убыточная'),
], loc='upper left', fontsize=8)

axes[1,1].hist(df[retail_col], bins=40, color='steelblue', edgecolor='white')
axes[1,1].set_title('Распределение розничной цены')
axes[1,1].set_xlabel('Розничная цена ($)')
axes[1,1].set_ylabel('Кол-во')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_plots.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Сохранено: {FIGURES_DIR}/eda_plots.png')

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax)
ax.set_title('Корреляционная матрица признаков')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/correlation.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Сохранено: {FIGURES_DIR}/correlation.png')

# ===
# 7. СПЛИТ: TRAIN / VAL / TEST  (70 / 15 / 15)
# ===

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

print('\n=== 7. Train / Val / Test сплит ===')
for name, XX, yy in [('Train', X_train, y_train),
                     ('Val',   X_val,   y_val),
                     ('Test',  X_test,  y_test)]:
    print(f'{name}: {len(XX):,} ({len(XX)/len(X)*100:.0f}%) | class 1: {yy.mean()*100:.1f}%')

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ===
# 8. МОДЕЛИ
# ===

print('\n=== 8. Модели ===')

# --- 8.1 Baseline: Logistic Regression (линейная) ---
print('\n-- 8.1 Logistic Regression (Baseline, линейная) --')
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
lr.fit(X_train_sc, y_train)

lr_pred_val  = lr.predict(X_val_sc)
lr_prob_val  = lr.predict_proba(X_val_sc)[:, 1]
lr_pred_test = lr.predict(X_test_sc)
lr_prob_test = lr.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, lr_pred_test, target_names=['Убыточная', 'Прибыльная']))
lr_f1  = f1_score(y_test, lr_pred_test, average='macro')
lr_auc = roc_auc_score(y_test, lr_prob_test)
print(f'F1-macro: {lr_f1:.4f} | ROC-AUC: {lr_auc:.4f}')

# --- 8.2 Random Forest (нелинейная) ---
print('\n-- 8.2 Random Forest (нелинейная) --')
rf = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred_val  = rf.predict(X_val)
rf_prob_val  = rf.predict_proba(X_val)[:, 1]
rf_pred_test = rf.predict(X_test)
rf_prob_test = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, rf_pred_test, target_names=['Убыточная', 'Прибыльная']))
rf_f1  = f1_score(y_test, rf_pred_test, average='macro')
rf_auc = roc_auc_score(y_test, rf_prob_test)
print(f'F1-macro: {rf_f1:.4f} | ROC-AUC: {rf_auc:.4f}')

# --- Сравнительная таблица ---
print('\n-- Сравнение моделей (Test set) --')
print(f'{"Модель":<25} {"F1-macro":>10} {"ROC-AUC":>10}')
print('-' * 47)
print(f'{"Logistic Regression":<25} {lr_f1:>10.4f} {lr_auc:>10.4f}')
print(f'{"Random Forest":<25} {rf_f1:>10.4f} {rf_auc:>10.4f}')

# --- Confusion Matrix обеих моделей ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, lr_pred_test, ax=axes[0],
    display_labels=['Убыточная', 'Прибыльная'],
    colorbar=False, cmap='Blues')
axes[0].set_title('Logistic Regression (Baseline)')

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred_test, ax=axes[1],
    display_labels=['Убыточная', 'Прибыльная'],
    colorbar=False, cmap='Greens')
axes[1].set_title('Random Forest')

plt.suptitle('Confusion Matrix — сравнение моделей (Test set)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Сохранено: {FIGURES_DIR}/confusion_matrix.png')

# --- Feature Importance (Random Forest) ---
fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
fi.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Feature Importance — Random Forest')
ax.set_ylabel('Importance')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Сохранено: {FIGURES_DIR}/feature_importance.png')

# ===
# 9. ВЫВОДЫ
# ===
print('\n=== 9. Выводы ===')
print(f'Logistic Regression:  F1-macro = {lr_f1:.2f}, ROC-AUC = {lr_auc:.2f}')
print(f'Random Forest:        F1-macro = {rf_f1:.2f}, ROC-AUC = {rf_auc:.2f}')
print()
print('Обе модели дают метрики близкие к случайному угадыванию (0.50).')
print('Это подтверждает: проблема в данных, а не в коде.')
print('Признаки condition, retail_price, sales_channel, size, quarter')
print('не дают достаточно информации для предсказания прибыльности.')
print()
print('В CP2 планируется:')
print('  - более глубокий feature engineering')
print('  - подбор гиперпараметров (GridSearch)')