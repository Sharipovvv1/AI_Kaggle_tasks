# Решения практических заданий Kaggle

В репозитории представлены блокноты с решениями трёх соревнований.

## 1. Linear Regression (linear-regression-competition-2026)
* **Файл:** `linear_regression.ipynb`
* **Задача:** Регрессия (прогнозирование стоимости недвижимости за единицу площади).
* **Подход:** Линейная модель RidgeCV с подбором коэффициента регуляризации на 5 фолдах кросс-валидации. Добавлены полиномиальные признаки (PolynomialFeatures) и нормализация через RobustScaler.
* **Результат на Kaggle:** Private Score 63.28

## 2. Mental Health Prediction (mental-health-prediction-2026)
* **Файл:** `mental_health.ipynb`
* **Задача:** Бинарная классификация (определение наличия депрессии по ответам анкеты).
* **Подход:** Логистическая регрессия (LogisticRegression, solver='lbfgs') с сохранением структурных нулей и One-Hot Encoding категориальных признаков.
* **Результат на Kaggle:** Private Score 0.98907 (Accuracy ~98.9%)

## 3. Vacancies Salary Prediction (vacancies-2026)
* **Файл:** `vacancies.ipynb`
* **Задача:** Регрессия (прогнозирование зарплаты вакансии по данным с HeadHunter).
* **Подход:** Ансамбль моделей: текстовая модель TF-IDF + Ridge по очищенному описанию вакансий и градиентный бустинг CatBoostRegressor на 5 фолдах с извлечением грейдов (Junior/Middle/Senior).
* **Результат на Kaggle:** Private MAPE 0.198 (ошибка ~19.8%)
