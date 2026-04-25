# ML Project  - Предсказание прибыльности перепродажи кроссовок Air Jordan

**Студент:** Лукьянов Ярослав Владимирович

**Группа:** БИВ 238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

По характеристикам кроссовок и условиям продажи предсказать, будет ли сделка перепродажи прибыльной.

**Задача:** Бинарная классификация

**Датасет:** [Air Jordan Sneaker Market & Resale Data (2023–2026)](https://www.kaggle.com/datasets/abdullahmeo/air-jordan-sneaker-market-and-resale-data2023-2026?resource=download)  - реальные данные о розничных ценах и ценах перепродажи кроссовок Air Jordan

**Целевая переменная:** `is_profitable = 1` если `resale_price > retail_price`, иначе `0`

**Целевая метрика:** F1 -macro (устойчива к дисбалансу классов) + ROC -AUC (для сравнения моделей)

**Почему F1 -macro, а не Accuracy:** При дисбалансе классов (61% убыточных / 39% прибыльных) модель «всё = 0» даст 61% accuracy, но будет бесполезна. F1 -macro одинаково штрафует за ошибки в обоих классах.


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы датасета
├── reports
│   └── figures                 # Графики: EDA, корреляция, confusion matrix
├── src
│   └── jordan_resale_cp1.py    # Основной скрипт: EDA, очистка, feature engineering, модели
├── requirements.txt            # Зависимости с версиями
└── README.md
```


## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo -name>

# 2. Создать виртуальное окружение
python  -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install  -r requirements.txt

# 4. Скачать датасет с Kaggle и положить в data/raw/
# https://www.kaggle.com/datasets/abdullahmeo/air -jordan -sneaker -market -and -resale -data2023 -2026

# 5. Запустить из корня репозитория
python src/jordan_resale_cp1.py
```

После запуска в папке `reports/figures/` появятся графики:
 - `eda_plots.png`  - баланс классов, распределение прибыли, scatter, гистограмма цен
 - `correlation.png`  - корреляционная матрица признаков
 - `confusion_matrix.png`  - матрицы ошибок обеих моделей
 - `feature_importance.png`  - важность признаков (Random Forest)


## Данные

 - `data/raw/`  - исходный файл `jordan_market_dataset_2026.csv`


**Описание датасета:**

| Параметр | Значение |
|---|--|
| Строк | 5 000 |
| Столбцов | 10 |
| Период | 2023–2026 |
| Источник | Kaggle |

**Ключевые столбцы:**

| Столбец | Описание |
|---|----|
| `shoe_model` | Модель кроссовка |
| `colorway` | Расцветка |
| `condition` | Состояние (новые/бу) |
| `retail_price_usd` | Розничная цена ($) |
| `resale_price_usd` | Цена перепродажи ($)  - **удалён (утечка)** |
| `sales_channel` | Канал продажи |
| `days_in_inventory` | Дней в наличии до продажи |
| `profit_margin_usd` | Маржа ($)  - **удалён (утечка)** |

**Feature Engineering (новые признаки):**

| Признак | Описание |
|------|--------|
| `quarter` | Квартал продажи  - учитывает сезонные циклы спроса |
| `is_common_size` | 1 если размер 8–11 US (ходовые ликвидные размеры) |
| `size_category` | Toddler/GS, Women, Men  - разная динамика спроса |

**Предотвращение data leakage:**
 - `resale_price_usd` удалён  - напрямую раскрывает ответ
 - `profit_margin_usd` удалён  - это `resale  - retail`, тот же ответ в другом виде
 - `profit` и `is_profitable` исключены
 - `StandardScaler` обучается только на train


## Результаты

### CP1  - Сравнение моделей (Test set, 70/15/15)

| Модель | F1-macro | ROC-AUC | Тип |
|--------|----------|---------|-----|
| Logistic Regression | 0.51 | 0.56 | Линейная (Baseline) |
| Random Forest | 0.94 | 0.97 | Нелинейная |

**Вывод:** Logistic Regression даёт метрики близкие к случайному угадыванию (0.50)  - линейные зависимости в данных слабые. Random Forest улавливает нелинейные паттерны и показывает значительно лучший результат. В CP2 будут внесенны улучшения.
