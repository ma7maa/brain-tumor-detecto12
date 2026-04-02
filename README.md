# Brain Tumor Detector — YOLOv8 + Streamlit

تطبيق ويب لمساعدة بصرية على اكتشاف مناطق مشبوهة في صور تشبه MRI باستخدام **YOLOv8**، مع واجهة **Streamlit** ثنائية اللغة (عربي / English).

## المزايا

- رفع صورة (JPG / PNG / WebP) أو **التقاط صورة من الكاميرا**
- تعديل **عتبة الثقة (Confidence)** و **IoU** من الشريط الجانبي
- عرض **الصورة الأصلية** و**النتيجة المُعلّمة** جنباً إلى جنب
- جدول **الفئة والثقة** لكل كشف، و**عدد الكشوفات**
- **تحميل** الصورة الناتجة كـ PNG
- تبويب **عن المشروع** مع عرض **metrics.json** إن وُجد (بعد التدريب)
- متغير بيئة **`BRAIN_TUMOR_MODEL`**: مسار ملف الأوزان إذا لم يكن اسمه `last.pt`

## التشغيل

```bash
pip install -r requirements.txt
```

ضع ملف الأوزان **`last.pt`** في جذر المشروع (نفس مجلد `app.py`)، ثم:

```bash
streamlit run app.py
```

**وضع التصنيف VGG16 في الواجهة:** بعد التدريب ضع `models/vgg16_brain_best.keras` و`models/class_indices.json`، وثبّت TensorFlow في نفس بيئة Streamlit (`pip install tensorflow`)، ثم من الشريط الجانبي اختر **VGG16 — تصنيف الصورة**.
متغيرات اختيارية: `BRAIN_VGG_MODEL`، `BRAIN_VGG_CLASSES`.

بديل متوافق مع الإصدارات السابقة:

```bash
streamlit run TUMOUR1.py
```

## تدريب موديل التصنيف VGG16 على الداتا المحلية

بعد فك `archive.zip` إلى `data/brain_mri/`:

```bash
pip install -r requirements-train.txt
python train_vgg16_brain_mri.py
```

**Windows (Python من Microsoft Store):** تثبيت TensorFlow أحياناً يفشل بسبب **مسارات طويلة**. الحل العملي: بيئة افتراضية في مسار قصير، مثلاً:

```powershell
C:\Users\M7MD\AppData\Local\Microsoft\WindowsApps\python3.11.exe -m venv C:\bt\brainvenv
C:\bt\brainvenv\Scripts\pip install tensorflow scikit-learn tqdm
cd C:\Users\M7MD\Downloads\brain-tumor-detector-main\brain-tumor-detector-main
C:\bt\brainvenv\Scripts\python.exe train_vgg16_brain_mri.py
```

أو تفعيل **Long Paths** في ويندوز ثم إعادة `pip install -r requirements-train.txt` على نفس بايثونك.

المخرجات: `models/vgg16_brain_best.keras`، `models/vgg16_brain_final.keras`، `models/class_indices.json`، `models/test_metrics.json`.  
لإعادة التدريب بدون إعادة تقسيم الصور: `python train_vgg16_brain_mri.py --skip-split`.

## ملفات التسليم والتوثيق

| الملف | الوصف |
|--------|--------|
| `PROJECT_REPORT_AR.md` | أقسام 3–8 جاهزة للتقرير (حدّث الروابط والأرقام) |
| `metrics.json.example` | انسخه إلى `metrics.json` واملأ نتائج التدريب |
| `data/README.md` | إرشادات البيانات وهيكل YOLO |

## التقنيات

- Python، **Ultralytics YOLOv8**، **Streamlit**، OpenCV، Pillow، NumPy، PyTorch

## تنبيه طبي

النتائج **مساعدة فقط** ولا تغني عن تشخيص طبي مهني.

## الترخيص

MIT (أو حسب الملف `LICENSE` إن وُجد).
