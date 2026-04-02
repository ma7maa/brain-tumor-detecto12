"""
Brain Tumor Detector — Streamlit + YOLOv8
"""
import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

from ultralytics import YOLO

MODEL_PATH = os.environ.get("BRAIN_TUMOR_MODEL", "last.pt")
VGG_MODEL_PATH = os.environ.get(
    "BRAIN_VGG_MODEL",
    str(Path("models") / "vgg16_brain_best.keras"),
)
CLASS_INDICES_PATH = os.environ.get(
    "BRAIN_VGG_CLASSES",
    str(Path("models") / "class_indices.json"),
)

STRINGS = {
    "ar": {
        "page_title": "كاشف أورام الدماغ",
        "subtitle": "رفع صورة MRI أو التقاطها — YOLO (كشف) أو VGG16 (تصنيف)",
        "mode_label": "نوع النموذج",
        "mode_yolo": "YOLOv8 — كشف مناطق",
        "mode_vgg": "VGG16 — تصنيف الصورة (4 فئات)",
        "vgg_result": "احتمالات التصنيف",
        "vgg_top": "الفئة الأرجح",
        "vgg_missing": "ملف VGG غير موجود",
        "vgg_hint": "شغّل `train_vgg16_brain_mri.py` أو ضع `{path}` ثم أعد التشغيل.",
        "vgg_tf": "ثبّت TensorFlow للتصنيف: `pip install tensorflow`",
        "vgg_classes_missing": "ملف الفئات غير موجود — توقع أرقام فقط",
        "upload_tab": "رفع صورة",
        "camera_tab": "الكاميرا",
        "about_tab": "عن المشروع",
        "sidebar_settings": "الإعدادات",
        "lang_label": "اللغة",
        "conf_label": "عتبة الثقة (Confidence)",
        "iou_label": "عتبة IoU",
        "show_boxes": "إظهار الصناديق والتسميات",
        "upload_prompt": "اختر صورة (JPG / PNG)",
        "camera_prompt": "التقط صورة",
        "original": "الصورة الأصلية",
        "result": "نتيجة الكشف",
        "detections": "الكشوفات",
        "no_detections": "لم يُعثر على أورام فوق عتبة الثقة الحالية.",
        "download": "تحميل الصورة المُعلّمة",
        "model_missing": "ملف النموذج غير موجود",
        "model_hint": "ضع ملف الأوزان `{path}` بجانب التطبيق، أو عيّن المتغير `BRAIN_TUMOR_MODEL` لمسار الملف.",
        "processing": "جاري التحليل…",
        "class_col": "الفئة",
        "conf_col": "الثقة",
        "count_label": "عدد الكشوفات",
        "about_body": """
هذا التطبيق يستخدم نموذج **YOLOv8** المدرّب على صور MRI لمساعدة في **إبراز المناطق المشبوهة** فقط،
ولا يغني عن التشخيص الطبي. للتفاصيل الكاملة راجع ملف `PROJECT_REPORT_AR.md`.
        """,
        "metrics_title": "مقاييس التدريب (من metrics.json)",
        "no_metrics": "لا يوجد ملف metrics.json — انسخ metrics.json.example واملأ الأرقام.",
    },
    "en": {
        "page_title": "Brain Tumor Detector",
        "subtitle": "Upload or capture an MRI — YOLO (detect) or VGG16 (classify)",
        "mode_label": "Model type",
        "mode_yolo": "YOLOv8 — region detection",
        "mode_vgg": "VGG16 — whole-image classification (4 classes)",
        "vgg_result": "Classification probabilities",
        "vgg_top": "Top prediction",
        "vgg_missing": "VGG model file not found",
        "vgg_hint": "Run `train_vgg16_brain_mri.py` or place weights at `{path}`.",
        "vgg_tf": "Install TensorFlow for VGG: `pip install tensorflow`",
        "vgg_classes_missing": "class_indices.json missing — showing class indices only",
        "upload_tab": "Upload",
        "camera_tab": "Camera",
        "about_tab": "About",
        "sidebar_settings": "Settings",
        "lang_label": "Language",
        "conf_label": "Confidence threshold",
        "iou_label": "IoU threshold",
        "show_boxes": "Show boxes & labels",
        "upload_prompt": "Choose an image (JPG / PNG)",
        "camera_prompt": "Take a photo",
        "original": "Original",
        "result": "Detection result",
        "detections": "Detections",
        "no_detections": "No objects above the current confidence threshold.",
        "download": "Download annotated image",
        "model_missing": "Model weights not found",
        "model_hint": "Place `{path}` next to the app, or set `BRAIN_TUMOR_MODEL` to the file path.",
        "processing": "Running inference…",
        "class_col": "Class",
        "conf_col": "Confidence",
        "count_label": "Detections",
        "about_body": """
This app uses a **YOLOv8** model trained on MRI images to **highlight suspicious regions** only.
It does not replace medical diagnosis. See `PROJECT_REPORT_AR.md` for full documentation.
        """,
        "metrics_title": "Training metrics (from metrics.json)",
        "no_metrics": "No metrics.json — copy metrics.json.example and fill in values.",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "ar")
    return STRINGS.get(lang, STRINGS["ar"]).get(key, key)


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


@st.cache_resource
def load_vgg_model(weights_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(weights_path, compile=False)


@st.cache_data
def load_class_indices(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run_vgg_inference(clf, image_rgb: Image.Image, class_meta: dict | None):
    img = image_rgb.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    preds = clf.predict(batch, verbose=0)[0]
    n = len(preds)
    if class_meta and "class_indices" in class_meta:
        inv = {v: k for k, v in class_meta["class_indices"].items()}
        labels = [inv.get(i, str(i)) for i in range(n)]
    else:
        labels = [str(i) for i in range(n)]
    pairs = list(zip(labels, preds.astype(float)))
    pairs.sort(key=lambda x: -x[1])
    return pairs


def numpy_bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_inference(model, image_rgb: Image.Image, conf: float, iou: float, show_labels: bool):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_rgb.save(tmp.name, format="JPEG", quality=95)
        path = tmp.name
    try:
        results = model.predict(
            path,
            conf=conf,
            iou=iou,
            verbose=False,
        )[0]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    plotted = results.plot(boxes=True, labels=show_labels, conf=show_labels)
    plotted_rgb = numpy_bgr_to_rgb(plotted)

    rows = []
    names = results.names or {}
    if results.boxes is not None and len(results.boxes):
        for b in results.boxes:
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            cf = float(b.conf[0]) if b.conf is not None else 0.0
            label = names.get(cls_id, str(cls_id))
            rows.append({"class": label, "confidence": round(cf, 4)})

    return plotted_rgb, rows


def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "ar"

    st.set_page_config(page_title="Brain Tumor Detector", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        .main-title { font-size: 2.1rem; font-weight: 700;
            background: linear-gradient(90deg, #0ea5e9, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; margin-bottom: 0.25rem; }
        .sub-title { text-align: center; color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    root = Path(__file__).resolve().parent
    weights = Path(MODEL_PATH)
    if not weights.is_absolute():
        weights = root / weights
    vgg_weights = Path(VGG_MODEL_PATH)
    if not vgg_weights.is_absolute():
        vgg_weights = root / vgg_weights
    class_indices_file = Path(CLASS_INDICES_PATH)
    if not class_indices_file.is_absolute():
        class_indices_file = root / class_indices_file

    with st.sidebar:
        st.session_state.lang = st.selectbox(
            t("lang_label"),
            options=["ar", "en"],
            format_func=lambda x: "العربية" if x == "ar" else "English",
            index=0 if st.session_state.lang == "ar" else 1,
        )
        mode = st.radio(
            t("mode_label"),
            options=["yolo", "vgg"],
            format_func=lambda x: t("mode_yolo") if x == "yolo" else t("mode_vgg"),
            horizontal=False,
        )
        st.markdown(f"### {t('sidebar_settings')}")
        if mode == "yolo":
            conf = st.slider(t("conf_label"), 0.05, 0.95, 0.25, 0.05)
            iou = st.slider(t("iou_label"), 0.1, 0.95, 0.45, 0.05)
            show_labels = st.checkbox(t("show_boxes"), value=True)
        else:
            conf = iou = 0.25
            show_labels = True
        st.caption("YOLO: " + ("OK" if weights.is_file() else f"Missing `{weights.name}`"))
        st.caption("VGG: " + ("OK" if vgg_weights.is_file() else f"Missing `{vgg_weights.name}`"))

    st.markdown(f'<p class="main-title">{t("page_title")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{t("subtitle")}</p>', unsafe_allow_html=True)

    tab_upload, tab_cam, tab_about = st.tabs(
        [t("upload_tab"), t("camera_tab"), t("about_tab")]
    )

    image_rgb = None

    with tab_upload:
        up = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            image_rgb = Image.open(up).convert("RGB")

    with tab_cam:
        cam = st.camera_input(t("camera_prompt"))
        if cam is not None:
            image_rgb = Image.open(cam).convert("RGB")

    with tab_about:
        st.markdown(t("about_body"))
        metrics_path = root / "metrics.json"
        st.subheader(t("metrics_title"))
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.json(data)
            except json.JSONDecodeError as e:
                st.error(str(e))
        else:
            st.caption(t("no_metrics"))

    if image_rgb is None:
        return

    if mode == "yolo":
        if not weights.is_file():
            st.error(f"**{t('model_missing')}:** `{weights}`")
            st.info(t("model_hint").format(path=weights.name))
            return
        try:
            model = load_model(str(weights.resolve()))
        except Exception as e:
            st.error(f"Failed to load YOLO: {e}")
            return
        with st.spinner(t("processing")):
            plotted_rgb, rows = run_inference(model, image_rgb, conf, iou, show_labels)
        c1, c2 = st.columns(2)
        with c1:
            st.image(image_rgb, caption=t("original"), use_column_width=True)
        with c2:
            st.image(plotted_rgb, caption=t("result"), use_column_width=True)
        st.metric(t("count_label"), len(rows))
        if rows:
            st.subheader(t("detections"))
            st.dataframe(rows, use_column_width=True)
        else:
            st.warning(t("no_detections"))
        buf = io.BytesIO()
        Image.fromarray(plotted_rgb).save(buf, format="PNG")
        st.download_button(
            label=t("download"),
            data=buf.getvalue(),
            file_name="brain_tumor_detection.png",
            mime="image/png",
        )
        return

    if not vgg_weights.is_file():
        st.error(f"**{t('vgg_missing')}:** `{vgg_weights}`")
        st.info(t("vgg_hint").format(path=vgg_weights.name))
        return

    try:
        clf = load_vgg_model(str(vgg_weights.resolve()))
    except ImportError:
        st.error(t("vgg_tf"))
        return
    except Exception as e:
        st.error(f"VGG load error: {e}")
        return

    meta = load_class_indices(str(class_indices_file))
    if meta is None:
        st.warning(t("vgg_classes_missing"))

    with st.spinner(t("processing")):
        pairs = run_vgg_inference(clf, image_rgb, meta)

    st.image(image_rgb, caption=t("original"), use_column_width=True)
    top_label, top_p = pairs[0]
    st.metric(t("vgg_top"), f"{top_label} ({100.0 * top_p:.2f}%)")
    st.subheader(t("vgg_result"))
    df = pd.DataFrame(pairs, columns=[t("class_col"), t("conf_col")])
    st.dataframe(df, use_column_width=True)
    chart_df = pd.DataFrame({t("class_col"): [x[0] for x in pairs], "p": [x[1] for x in pairs]})
    st.bar_chart(chart_df.set_index(t("class_col")))


if __name__ == "__main__":
    main()
