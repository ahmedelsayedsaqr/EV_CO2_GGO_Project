# مشروع التنبؤ بانبعاثات ثاني أكسيد الكربون للمركبات (EV_CO2_GGO_Project)

## نظرة عامة

يهدف هذا المشروع إلى بناء نموذج تعلم آلي للتنبؤ بانبعاثات ثاني أكسيد الكربون (CO2) للمركبات، مع التركيز على تحسين أداء النموذج باستخدام خوارزمية **Greylag Goose Optimization (GGO)** لضبط المعاملات الفائقة لنموذج **Multi-Layer Perceptron (MLP)**.

تم استلهام المنهجية من البحث الأكاديمي المرفق: [Enhancing CO2 emissions prediction for electric vehicles using Greylag Goose Optimization and machine learning](https://www.nature.com/articles/s41598-025-99472-0)

## هيكل المشروع

```
EV_CO2_GGO_Project/
│
├─ data/
│   ├─ Fuel_Consumption_2000-2022.csv   ← ملف البيانات الخام
│   └─ README.txt                        ← تعليمات حول البيانات
│
├─ scripts/
│   ├─ data_prep.py                      ← دوال تجهيز البيانات (تحميل، تقسيم، معالجة مسبقة)
│   ├─ ggo.py                            ← تطبيق خوارزمية Greylag Goose Optimization
│   ├─ optimize.py                       ← تشغيل GGO وحفظ أفضل المعاملات الفائقة
│   ├─ train_mlp.py                      ← تدريب النموذج النهائي باستخدام المعاملات المحسّنة
│   └─ evaluate.py                       ← تقييم النموذج وحفظ المقاييس والرسومات
│
├─ notebooks/
│   └─ analysis.ipynb                    ← نوتبوك Jupyter يوضح سير العمل خطوة بخطوة
│
├─ results/
│   ├─ figures/                          ← الرسومات البيانية الناتجة (مثل Actual vs. Predicted)
│   ├─ metrics.json                      ← ناتج التقييم النهائي للنموذج
│   └─ best_params.json                  ← أفضل المعاملات الفائقة التي وجدتها GGO
│
├─ requirements.txt                       ← المكاتب المطلوبة لتشغيل المشروع
└─ README.md                              ← هذا الملف
```

## المتطلبات (Requirements)

لتشغيل المشروع، تحتاج إلى تثبيت المكتبات التالية:

```bash
pip install -r requirements.txt
```

## طريقة التشغيل (How to Run)

يمكن تشغيل المشروع بثلاث طرق رئيسية:

### 1. التشغيل عبر النوتبوك (Recommended)

افتح ملف `notebooks/analysis.ipynb` في بيئة Jupyter Notebook أو JupyterLab. يحتوي النوتبوك على جميع الخطوات اللازمة لتشغيل المشروع بالترتيب، بما في ذلك:
1.  تثبيت المتطلبات.
2.  تشغيل خوارزمية GGO (ملف `optimize.py`).
3.  تدريب النموذج النهائي (ملف `train_mlp.py`).
4.  تقييم النموذج وعرض النتائج (ملف `evaluate.py`).

### 2. التشغيل عبر سطر الأوامر (Sequential)

يمكنك تشغيل الخطوات بشكل تسلسلي من سطر الأوامر:

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. تشغيل خوارزمية GGO لتحسين المعاملات الفائقة
python scripts/optimize.py

# 3. تدريب النموذج النهائي وتقييمه
python scripts/evaluate.py
```

### 3. استعراض النتائج

بعد التشغيل، ستجد النتائج في مجلد `results/`:
-   **`results/best_params.json`**: أفضل المعاملات الفائقة التي وجدتها GGO.
-   **`results/metrics.json`**: مقاييس الأداء النهائية للنموذج (R2, RMSE, MAE).
-   **`results/figures/actual_vs_predicted.png`**: الرسم البياني للمقارنة بين القيم الفعلية والمتوقعة.

## المنهجية المتبعة

1.  **تجهيز البيانات (`data_prep.py`):**
    -   تحميل البيانات واختيار الميزات ذات الصلة (حجم المحرك، عدد الأسطوانات، استهلاك الوقود، فئة المركبة، نوع ناقل الحركة، نوع الوقود).
    -   تطبيق **StandardScaler** على الميزات الرقمية و **OneHotEncoder** على الميزات الفئوية.
    -   تقسيم البيانات إلى مجموعتي تدريب واختبار.

2.  **التحسين باستخدام GGO (`ggo.py` و `optimize.py`):**
    -   تم تطبيق خوارزمية **Greylag Goose Optimization (GGO)** لتحسين المعاملات الفائقة لنموذج MLP (مثل معدل التعلم وأحجام الطبقات المخفية).
    -   دالة الهدف (Objective Function) هي تقليل **جذر متوسط مربع الخطأ (RMSE)** عبر التحقق المتقاطع (Cross-Validation).

3.  **التدريب والتقييم (`train_mlp.py` و `evaluate.py`):**
    -   يتم تدريب نموذج MLP النهائي باستخدام أفضل المعاملات الفائقة التي وجدتها GGO.
    -   يتم تقييم النموذج على مجموعة الاختبار باستخدام مقاييس الانحدار القياسية وحفظها في `metrics.json`.
    -   يتم إنشاء رسم بياني يوضح مدى دقة التنبؤ.
