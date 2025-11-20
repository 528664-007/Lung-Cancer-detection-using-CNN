# Full corrected Colab-ready pipeline (single cell)
# - Improved Grad-CAM (smoothing + better normalization)
# - Optional Transfer-Learning training (MobileNetV2)
# - Fixed ZIP creation & download
# - Enhanced UI (preview, regen, merge PDFs, CSV)
# NOTE: This cell may take several minutes to run (installs + TF import)

# ----------------------------
# Installs
# ----------------------------
!pip install --quiet reportlab opencv-python-headless==4.7.0.72 pillow tqdm pypdf2 tensorflow keras ipywidgets

# ----------------------------
# Imports
# ----------------------------
import os, io, zipfile, math, base64
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tqdm import tqdm
from PyPDF2 import PdfMerger  # installed above

# ----------------------------
# Settings / Constants
# ----------------------------
MODEL_WEIGHTS_PATH = None   # optional: pretrained .h5 path
IMAGE_SAVE_DIR = "results_futuristic"
TARGET_SIZE = (224, 224)    # MobileNetV2 default-like; changed from 256->224 for transfer learning
THRESHOLD_RATIO_DEFAULT = 0.60
MIN_AREA_DEFAULT = 30
ALPHA = 0.45
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = 2
NUMBER_COLOR = (255, 255, 255)
COLORMAP = cv2.COLORMAP_JET
SMOOTHGRAD_SAMPLES = 8      # number of noisy samples to average for smoother CAM
SMOOTHGRAD_STD = 0.03

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("uploaded_zip", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# ----------------------------
# Helper: download link
# ----------------------------
def file_download_link(path, label=None):
    label = label or os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return HTML(f'<a download="{os.path.basename(path)}" href="data:application/octet-stream;base64,{b64}">{label}</a>')

# ----------------------------
# Build / Load classifier: Transfer Learning (MobileNetV2)
# ----------------------------
def build_classifier_mobilenetv2(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), dropout_rate=0.3):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, alpha=1.0)
    base.trainable = False  # start frozen
    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model, base

# Try to use fine-tuned or demo model
try:
    classifier, backbone = build_classifier_mobilenetv2()
    if MODEL_WEIGHTS_PATH and os.path.exists(MODEL_WEIGHTS_PATH):
        classifier.load_weights(MODEL_WEIGHTS_PATH)
except Exception as e:
    # fallback minimal model (should not normally happen)
    print("Transfer model build failed; building lightweight fallback.", e)
    def build_fallback(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)):
        inputs = Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)
        m = Model(inputs,out); m.compile('adam','binary_crossentropy',['accuracy'])
        return m
    classifier = build_fallback()

# ----------------------------
# Optional training function (call when you have labeled data)
# Directory structure expected:
# data_dir/
#   train/
#     pos/
#     neg/
#   val/
#     pos/
#     neg/
# ----------------------------
def train_transfer_learning(data_dir, out_model_path="trained_models/best_model.h5",
                            input_size=TARGET_SIZE, batch_size=16, epochs=20, fine_tune_at=100):
    """Train MobileNetV2 with ImageDataGenerator. Returns trained model path."""
    tr_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not (os.path.exists(tr_dir) and os.path.exists(val_dir)):
        raise ValueError("Expected train/ and val/ subfolders under data_dir.")
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen_val = ImageDataGenerator(rescale=1./255)
    train_gen = datagen_train.flow_from_directory(tr_dir, target_size=input_size, batch_size=batch_size, class_mode='binary')
    val_gen = datagen_val.flow_from_directory(val_dir, target_size=input_size, batch_size=batch_size, class_mode='binary')
    model, base = build_classifier_mobilenetv2(input_shape=(input_size[0], input_size[1],3))
    # callbacks
    es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    cp = ModelCheckpoint(out_model_path, monitor='val_loss', save_best_only=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    # initial training (top only)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[es,cp,rl])
    # fine-tune some layers
    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < fine_tune_at:
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=epochs//2, callbacks=[es,cp,rl])
    return out_model_path

# ----------------------------
# Improved Grad-CAM with SmoothGrad averaging
# ----------------------------
def compute_gradcam_smooth(model, img_array_exp, last_conv_layer=None, nsamples=SMOOTHGRAD_SAMPLES, std=SMOOTHGRAD_STD):
    """Compute Grad-CAM by averaging over noisy samples (SmoothGrad-ish)."""
    # pick last conv if not given
    if last_conv_layer is None:
        conv_names = [ly.name for ly in model.layers if isinstance(ly, tf.keras.layers.Conv2D)]
        if not conv_names:
            raise ValueError("No Conv2D layers found in model.")
        last_conv_layer = conv_names[-1]
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer).output, model.output])
    cams = []
    base_input = img_array_exp.astype(np.float32)
    for i in range(max(1, nsamples)):
        jitter = np.random.normal(0, std, size=base_input.shape).astype(np.float32)
        noisy = np.clip(base_input + jitter, 0.0, 1.0)
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(noisy)
            # for binary, target is predicted prob of class 1
            target = preds[:, 0]
        grads = tape.gradient(target, conv_outputs)[0]  # shape (h,w,channels)
        weights = tf.reduce_mean(grads, axis=(0,1))
        conv = conv_outputs[0].numpy()
        cam = np.sum(conv * weights.numpy()[None,None,:], axis=-1)
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / (np.max(cam) + 1e-8)
        cams.append(cam)
    cam_avg = np.mean(np.stack(cams, axis=0), axis=0)
    cam_avg = cam_avg / (np.max(cam_avg) + 1e-8)
    return cam_avg

# ----------------------------
# Heuristic lung mask (same as earlier)
# ----------------------------
def auto_lung_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ge = clahe.apply(gray)
    blurred = cv2.GaussianBlur(ge, (5,5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7,7), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return closed
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, cnts_sorted, -1, 255, -1)
    return mask

# ----------------------------
# Preprocess and visualization helpers
# ----------------------------
def preprocess_image_cv(img_bgr, target_size=TARGET_SIZE):
    resized = cv2.resize(img_bgr, target_size)
    rgb_for_model = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = img_to_array(rgb_for_model) / 255.0
    array_exp = np.expand_dims(array, axis=0).astype(np.float32)
    pil_rgb = Image.fromarray(rgb_for_model)
    return resized, pil_rgb, array_exp

def detect_hotspots_from_heatmap(heatmap_uint8, threshold_ratio=0.6, min_area=30, lung_mask=None):
    thr_val = int(threshold_ratio * np.max(heatmap_uint8))
    if thr_val <= 0:
        thr_val = 1
    _, hm_thresh = cv2.threshold(heatmap_uint8, thr_val, 255, cv2.THRESH_BINARY)
    if lung_mask is not None:
        if lung_mask.shape != hm_thresh.shape:
            lung_mask = cv2.resize(lung_mask, (hm_thresh.shape[1], hm_thresh.shape[0]))
        hm_thresh = cv2.bitwise_and(hm_thresh, lung_mask)
    cnts, _ = cv2.findContours(hm_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(max(8, r))
        hotspots.append((center, radius, int(area)))
    hotspots.sort(key=lambda x: x[2], reverse=True)
    return hotspots, hm_thresh

def draw_hotspots_on_image(hotspots, image_bgr, text=None):
    img_copy = image_bgr.copy()
    for idx, (center, radius, _) in enumerate(hotspots, start=1):
        cv2.circle(img_copy, center, radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
        cv2.circle(img_copy, center, max(2, radius//10), (255,255,255), -1)
        label_pt = (max(0, center[0]-radius), max(15, center[1]-radius-5))
        cv2.putText(img_copy, str(idx), label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, NUMBER_COLOR, 2, cv2.LINE_AA)
    if text:
        cv2.putText(img_copy, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img_copy

def hotspot_statistics(hotspots, heatmap_uint8):
    stats = []
    total_pixels = heatmap_uint8.size
    for idx, (center, radius, area) in enumerate(hotspots, start=1):
        mask = np.zeros_like(heatmap_uint8, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        pixels = heatmap_uint8[mask == 255]
        mean_intensity = float(np.mean(pixels)) if pixels.size else 0.0
        peak_intensity = int(np.max(pixels)) if pixels.size else 0
        coverage_percent = (area / total_pixels) * 100 if total_pixels>0 else 0
        stats.append({
            "Hotspot": idx,
            "CenterX": center[0],
            "CenterY": center[1],
            "Radius(px)": radius,
            "Area(px)": area,
            "Mean Intensity": round(mean_intensity, 2),
            "Peak Intensity": peak_intensity,
            "Coverage %": round(coverage_percent, 4)
        })
    return pd.DataFrame(stats)

def compute_severity_score(df_stats):
    if df_stats.empty: return 0
    coverage_sum = float(df_stats["Coverage %"].sum())
    mean_int = float(df_stats["Mean Intensity"].mean())
    n = len(df_stats)
    score = coverage_sum*2 + mean_int/2 + 10*math.sqrt(n)
    return max(0, min(100, int(score)))

# ----------------------------
# PDF Report generator
# ----------------------------
def generate_pdf_report(image_name, original_img_path, overlay_img_path, mask_img_path, df_stats, severity_score, out_pdf_path):
    c = canvas.Canvas(out_pdf_path, pagesize=landscape(A4))
    W, H = landscape(A4)
    title = f"AI Grad-CAM Analysis Report: {image_name}"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, H - 40, title)
    c.setFont("Helvetica", 10)
    c.drawString(30, H - 58, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    gap = 20
    img_w = (W - 4*gap) / 3
    img_h = img_w * 0.66
    y_top = H - 120
    orig = ImageReader(original_img_path)
    c.drawImage(orig, gap, y_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True)
    c.drawString(gap, y_top - img_h - 12, "Original image")
    mask = ImageReader(mask_img_path)
    c.drawImage(mask, 2*gap + img_w, y_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True)
    c.drawString(2*gap + img_w, y_top - img_h - 12, "Lung mask")
    ov = ImageReader(overlay_img_path)
    c.drawImage(ov, 3*gap + 2*img_w, y_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True)
    c.drawString(3*gap + 2*img_w, y_top - img_h - 12, f"Grad-CAM overlay - Severity: {severity_score}/100")
    table_y = y_top - img_h - 50
    c.setFont("Helvetica-Bold", 11)
    c.drawString(gap, table_y, "Hotspot analytics (top rows):")
    c.setFont("Helvetica", 9)
    start_y = table_y - 16
    max_rows = 10
    col_x = [gap, gap + 60, gap + 120, gap + 200, gap + 300, gap + 380]
    headers = ["#","Center","Area(px)","Radius(px)","Mean Int","Coverage%"]
    for i, h in enumerate(headers):
        c.drawString(col_x[i], start_y, h)
    for ri, row in df_stats.head(max_rows).iterrows():
        y = start_y - 14 * (ri+1)
        c.drawString(col_x[0], y, str(int(row["Hotspot"])))
        c.drawString(col_x[1], y, f"{int(row['CenterX'])},{int(row['CenterY'])}")
        c.drawString(col_x[2], y, str(int(row["Area(px)"])))
        c.drawString(col_x[3], y, str(int(row["Radius(px)"])))
        c.drawString(col_x[4], y, str(row["Mean Intensity"]))
        c.drawString(col_x[5], y, str(row["Coverage %"]))
    c.showPage()
    c.save()

# ----------------------------
# Single-image pipeline (improved)
# ----------------------------
def process_single_image(image_path, threshold_ratio=THRESHOLD_RATIO_DEFAULT, min_area=MIN_AREA_DEFAULT, use_lung_mask=True, alpha=ALPHA, colormap=COLORMAP):
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    orig_bgr = cv2.resize(orig_bgr, TARGET_SIZE)
    lung_mask = auto_lung_mask(orig_bgr) if use_lung_mask else None
    resized_bgr, _, img_array_exp = preprocess_image_cv(orig_bgr, target_size=TARGET_SIZE)
    pred = float(classifier.predict(img_array_exp, verbose=0)[0][0])
    label = "🔴 Abnormal (Likely)" if pred > 0.5 else "🟢 Normal (Likely)"
    annot_text = f"{label} ({pred*100:.2f}%)"
    cam = compute_gradcam_smooth(classifier, img_array_exp, nsamples=SMOOTHGRAD_SAMPLES, std=SMOOTHGRAD_STD)
    cam_resized = cv2.resize(cam, (TARGET_SIZE[1], TARGET_SIZE[0]))  # ensure same shape (w,h)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(cam_uint8, colormap)
    overlay_bgr = cv2.addWeighted(resized_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
    hotspots, heatmask = detect_hotspots_from_heatmap(cam_uint8, threshold_ratio=threshold_ratio, min_area=min_area, lung_mask=lung_mask)
    marked_overlay = draw_hotspots_on_image(hotspots, overlay_bgr, text=annot_text)
    # mask visualization for PDF (convert single-channel mask or zeros)
    mask_vis = cv2.cvtColor(lung_mask, cv2.COLOR_GRAY2BGR) if lung_mask is not None else np.zeros_like(resized_bgr)
    # produce filled number overlay too
    mask_filled = np.zeros_like(resized_bgr, dtype=np.uint8)
    for (center, radius, _) in hotspots:
        cv2.circle(mask_filled, center, radius, CIRCLE_COLOR, -1)
    filled_overlay = cv2.addWeighted(resized_bgr, 0.6, mask_filled, 0.4, 0)
    filled_with_numbers = draw_hotspots_on_image(hotspots, filled_overlay.copy(), text=annot_text)
    # save outputs
    base = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = os.path.join(IMAGE_SAVE_DIR, f"{base}_{timestamp}")
    orig_path = out_prefix + "_orig.png"
    mask_path = out_prefix + "_mask.png"
    overlay_path = out_prefix + "_overlay.png"
    heatmap_path = out_prefix + "_heatmap.png"
    filled_path = out_prefix + "_filled.png"
    cv2.imwrite(orig_path, resized_bgr)
    cv2.imwrite(mask_path, mask_vis)
    cv2.imwrite(overlay_path, marked_overlay)
    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(filled_path, filled_with_numbers)
    df_stats = hotspot_statistics(hotspots, cam_uint8)
    severity_score = compute_severity_score(df_stats)
    pdf_path = out_prefix + "_report.pdf"
    generate_pdf_report(base, orig_path, overlay_path, mask_path, df_stats, severity_score, pdf_path)
    return {
        "orig_path": orig_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "heatmap_path": heatmap_path,
        "filled_path": filled_path,
        "pdf_path": pdf_path,
        "df_stats": df_stats,
        "severity_score": severity_score,
        "summary_row": {
            "image": base,
            "timestamp": timestamp,
            "prediction": pred,
            "label": label,
            "confidence": pred*100,
            "n_hotspots": len(hotspots),
            "severity_score": severity_score
        }
    }

# ----------------------------
# Batch processing and fixed ZIP creation
# ----------------------------
def process_batch_images(image_paths, threshold_ratio=THRESHOLD_RATIO_DEFAULT, min_area=MIN_AREA_DEFAULT, use_lung_mask=True, alpha=ALPHA, colormap=COLORMAP):
    all_summaries = []
    detailed_stats = {}
    for p in image_paths:
        try:
            out = process_single_image(p, threshold_ratio=threshold_ratio, min_area=min_area, use_lung_mask=use_lung_mask, alpha=alpha, colormap=colormap)
            all_summaries.append(out["summary_row"])
            detailed_stats[p] = out
            print(f"Saved outputs for {Path(p).stem}")
        except Exception as e:
            print(f"Failed {p}: {e}")
    df_summary = pd.DataFrame(all_summaries)
    csv_path = os.path.join(IMAGE_SAVE_DIR, "batch_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    # create zip properly: add files with paths relative to IMAGE_SAVE_DIR
    zip_out = os.path.join(IMAGE_SAVE_DIR, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(IMAGE_SAVE_DIR):
            for f in files:
                full = os.path.join(root, f)
                # avoid including the zip itself if run multiple times
                if full == zip_out: continue
                arcname = os.path.relpath(full, IMAGE_SAVE_DIR)
                zf.write(full, arcname)
    return df_summary, detailed_stats, zip_out

# ----------------------------
# UI: file upload + controls + preview + merge CSV/PDF
# ----------------------------
upload_btn = widgets.FileUpload(accept=".png,.jpg,.jpeg,.zip", multiple=True)
threshold_slider = widgets.FloatSlider(value=THRESHOLD_RATIO_DEFAULT, min=0.05, max=0.99, step=0.01, description='Threshold')
minarea_slider = widgets.IntSlider(value=MIN_AREA_DEFAULT, min=1, max=1000, step=1, description='MinArea')
lung_mask_toggle = widgets.Checkbox(value=True, description='Use Lung Mask')
process_button = widgets.Button(description="Process Uploads", button_style='primary')
colormap_dropdown = widgets.Dropdown(options=['JET','HOT','BONE','VIRIDIS','PLASMA','INFERNO','COOL','SUMMER'], value='JET', description='Colormap')
alpha_slider = widgets.FloatSlider(value=ALPHA, min=0.0, max=1.0, step=0.01, description='Overlay α')
merge_pdfs_btn = widgets.Button(description='Merge All PDFs', button_style='info')
download_csv_btn = widgets.Button(description='Download CSV', button_style='success')
preview_selector = widgets.Dropdown(options=[], description='Preview Image')
regen_btn = widgets.Button(description='Regenerate Variant', button_style='primary')
out_box = widgets.Output(layout={'border': '1px solid black'})
preview_out = widgets.Output(layout={'border': '1px solid gray'})
_last_batch_results = {"df_summary": None, "detailed_stats": None, "zip_path": None}

# ----------------------------
# Handler
# ----------------------------
def handle_process(b):
    with out_box:
        clear_output()
        if not upload_btn.value:
            print("Please upload one or more image files (or a zip).")
            return
        saved_paths = []
        # write uploads to disk
        for fn, fileinfo in upload_btn.value.items():
            content = fileinfo['content']
            path = os.path.join("uploaded_images", fn)
            with open(path, "wb") as f:
                f.write(content)
            if fn.lower().endswith(".zip"):
                # extract zip into uploaded_zip
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall("uploaded_zip")
        # collect images from uploaded_images and uploaded_zip
        for root in ("uploaded_images", "uploaded_zip"):
            for rd, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith((".png",".jpg",".jpeg")):
                        saved_paths.append(os.path.join(rd, f))
        if not saved_paths:
            print("No image files found in uploads.")
            return
        print(f"Found {len(saved_paths)} images. Starting processing...")
        # set colormap constant
        cmap_name = colormap_dropdown.value.upper()
        cmap_attr = getattr(cv2, f'COLORMAP_{cmap_name}', cv2.COLORMAP_JET)
        df_summary, detailed_stats, zip_out = process_batch_images(
            saved_paths,
            threshold_ratio=threshold_slider.value,
            min_area=minarea_slider.value,
            use_lung_mask=lung_mask_toggle.value,
            alpha=alpha_slider.value,
            colormap=cmap_attr
        )
        print("\nBatch Summary:")
        display(df_summary)
        _last_batch_results["df_summary"] = df_summary
        _last_batch_results["detailed_stats"] = detailed_stats
        _last_batch_results["zip_path"] = zip_out
        preview_selector.options = sorted(list(detailed_stats.keys()))
        print(f"\nAll outputs zipped: {zip_out}")
        display(file_download_link(zip_out, "Download all results (ZIP)"))
        display(file_download_link(os.path.join(IMAGE_SAVE_DIR, "batch_summary.csv"), "Download summary CSV"))

process_button.on_click(handle_process)

# Merge PDFs handler (safely)
def handle_merge_pdfs(b):
    with out_box:
        clear_output()
        ds = _last_batch_results["detailed_stats"]
        if not ds:
            print("No batch results available. Run processing first.")
            return
        merger = PdfMerger()
        count = 0
        for v in ds.values():
            pdfp = v.get("pdf_path")
            if pdfp and os.path.exists(pdfp):
                merger.append(pdfp)
                count += 1
        if count == 0:
            print("No pdfs found.")
            return
        merged_path = os.path.join(IMAGE_SAVE_DIR, f"merged_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        merger.write(merged_path); merger.close()
        print(f"Merged {count} PDFs -> {merged_path}")
        display(file_download_link(merged_path, "Download merged PDF"))

merge_pdfs_btn.on_click(handle_merge_pdfs)

def handle_download_csv(b):
    with out_box:
        clear_output()
        if _last_batch_results["df_summary"] is None:
            print("No summary available.")
            return
        csv_path = os.path.join(IMAGE_SAVE_DIR, "batch_summary.csv")
        display(file_download_link(csv_path, "Download summary CSV"))

download_csv_btn.on_click(handle_download_csv)

def preview_image_change(change):
    with preview_out:
        clear_output()
        key = preview_selector.value
        if not key:
            return
        det = _last_batch_results["detailed_stats"].get(key)
        if det is None:
            print("Not found.")
            return
        img = cv2.imread(det["overlay_path"])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(rgb)
        plt.title(f"{Path(key).name}  |  Severity: {det['severity_score']}")
        plt.axis("off")
        plt.show()
        # show top hotspot table
        if not det["df_stats"].empty:
            display(det["df_stats"].head(10))
        display(file_download_link(det["pdf_path"], "Download image PDF report"))

preview_selector.observe(preview_image_change, names='value')

# Regenerate variant preview (no reprocessing)
def regenerate_variant(b):
    with preview_out:
        clear_output()
        key = preview_selector.value
        if not key:
            print("No image selected.")
            return
        det = _last_batch_results["detailed_stats"].get(key)
        if det is None:
            print("Selected image not found.")
            return
        # reconstruct orig and heatmap, blend with selected alpha and colormap
        orig = cv2.imread(det["orig_path"])
        heat = cv2.imread(det["heatmap_path"])
        if orig is None or heat is None:
            print("Required files missing.")
            return
        cam_gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
        cmap_name = colormap_dropdown.value.upper()
        cmap_attr = getattr(cv2, f'COLORMAP_{cmap_name}', cv2.COLORMAP_JET)
        recol = cv2.applyColorMap(cam_gray, cmap_attr)
        blended = cv2.addWeighted(orig, 1.0 - alpha_slider.value, recol, alpha_slider.value, 0)
        # draw same hotspots
        df = det["df_stats"]
        hotspots = []
        if not df.empty:
            for _, r in df.iterrows():
                hotspots.append(((int(r["CenterX"]), int(r["CenterY"])), int(r["Radius(px)"]), int(r["Area(px)"])))
        variant = draw_hotspots_on_image(hotspots, blended, text=f"Variant α={alpha_slider.value:.2f}")
        display(Image.fromarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)))
        # save variant temp
        tmp = os.path.join(IMAGE_SAVE_DIR, f"variant_{Path(key).stem}_{datetime.now().strftime('%H%M%S')}.png")
        cv2.imwrite(tmp, variant)
        display(file_download_link(tmp, "Download variant PNG"))

regen_btn.on_click(regenerate_variant)

# ----------------------------
# Final UI layout
# ----------------------------
ui = widgets.VBox([
    widgets.HTML("<h3>Futuristic Grad-CAM + Hotspot Analytics (Corrected)</h3>"),
    widgets.HBox([upload_btn, process_button]),
    widgets.HBox([threshold_slider, minarea_slider, lung_mask_toggle]),
    widgets.HBox([colormap_dropdown, alpha_slider]),
    widgets.HBox([merge_pdfs_btn, download_csv_btn, preview_selector, regen_btn]),
    out_box,
    preview_out
])
display(ui)

print("Ready — upload images (or a zip) and click 'Process Uploads'.")
print("If you have labeled training data, call `train_transfer_learning(data_dir)` to train a stronger model (see code comments).")
