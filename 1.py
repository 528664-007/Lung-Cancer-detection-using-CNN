# Full upgraded Colab pipeline: Grad-CAM + Hotspot Analytics + Interactive UI + PDF reports
# Paste into Colab and run (Runtime: GPU if available for speed)
!pip install reportlab opencv-python

# -------------------------
# Step 0: Imports
# -------------------------
import os
import io
import zipfile
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import img_to_array
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------
# Step 1: Settings & Params
# -------------------------
MODEL_WEIGHTS_PATH = None   # Optional: path to classifier weights (HDF5). If None, uses demo model.
SEGMENTATION_WEIGHTS_PATH = None  # Optional: path to UNet weights for lung segmentation (if you have)
IMAGE_SAVE_DIR = "results_futuristic"
TARGET_SIZE = (256, 256)
THRESHOLD_RATIO_DEFAULT = 0.60
MIN_AREA_DEFAULT = 30
ALPHA = 0.45
CIRCLE_COLOR = (0, 0, 255)      # BGR red
CIRCLE_THICKNESS = 2
NUMBER_COLOR = (255, 255, 255)  # white
COLORMAP = cv2.COLORMAP_JET

# Create directories
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("uploaded_zip", exist_ok=True)

# -------------------------
# Utilities: model + preprocessing
# -------------------------
def build_demo_classifier(input_shape=(256,256,3)):
    """Lightweight demo classifier - used if no weights provided."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build or load classifier
classifier = build_demo_classifier(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
if MODEL_WEIGHTS_PATH:
    classifier.load_weights(MODEL_WEIGHTS_PATH)

# -------------------------
# Lung Segmentation (fast heuristic)
# -------------------------
def auto_lung_mask(img_bgr, debug=False):
    """Fast lung mask heuristic"""
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

# -------------------------
# Preprocess image -> model input
# -------------------------
def preprocess_image_cv(img_bgr, target_size=TARGET_SIZE):
    resized = cv2.resize(img_bgr, target_size)
    rgb_for_model = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = img_to_array(rgb_for_model) / 255.0
    array_exp = np.expand_dims(array, axis=0)
    pil_rgb = Image.fromarray(rgb_for_model)
    return resized, pil_rgb, array_exp

# -------------------------
# Grad-CAM implementation
# -------------------------
def compute_gradcam(model, img_array_exp, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = [layer.name for layer in model.layers
                               if isinstance(layer, Conv2D)][-1]

    grad_model = Model(inputs=model.input,
                      outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array_exp)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

# -------------------------
# Hotspot detection & drawing
# -------------------------
def detect_hotspots_from_heatmap(heatmap_uint8, threshold_ratio=0.6, min_area=30, lung_mask=None):
    thr_val = int(threshold_ratio * np.max(heatmap_uint8))
    _, hm_thresh = cv2.threshold(heatmap_uint8, thr_val, 255, cv2.THRESH_BINARY)
    if lung_mask is not None:
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

# -------------------------
# Hotspot analytics & severity scoring
# -------------------------
def hotspot_statistics(hotspots, heatmap_uint8):
    stats = []
    total_pixels = heatmap_uint8.size
    for idx, (center, radius, area) in enumerate(hotspots, start=1):
        mask = np.zeros_like(heatmap_uint8, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        pixels = heatmap_uint8[mask == 255]
        if pixels.size == 0:
            mean_intensity = 0.0
            peak_intensity = 0
        else:
            mean_intensity = float(np.mean(pixels))
            peak_intensity = int(np.max(pixels))
        coverage_percent = (area / total_pixels) * 100
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
    df = pd.DataFrame(stats)
    return df

def compute_severity_score(df_stats):
    if df_stats.empty:
        return 0
    coverage_sum = float(df_stats["Coverage %"].sum())
    mean_int = float(df_stats["Mean Intensity"].mean())
    n = len(df_stats)
    score = coverage_sum*2 + mean_int/2 + 10*math.sqrt(n)
    score = max(0, min(100, int(score)))
    return score

# -------------------------
# PDF Report generation
# -------------------------
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

# -------------------------
# Single-image processing pipeline
# -------------------------
def process_single_image(image_path, threshold_ratio=THRESHOLD_RATIO_DEFAULT, min_area=MIN_AREA_DEFAULT, use_lung_mask=True):
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    orig_bgr = cv2.resize(orig_bgr, TARGET_SIZE)

    lung_mask = auto_lung_mask(orig_bgr) if use_lung_mask else None

    resized_bgr, _, img_array_exp = preprocess_image_cv(orig_bgr)
    pred = float(classifier.predict(img_array_exp)[0][0])
    confidence = pred * 100.0
    label = "🔴 Abnormal (Likely)" if pred > 0.5 else "🟢 Normal (Likely)"
    annot_text = f"{label} ({confidence:.2f}%)"

    cam = compute_gradcam(classifier, img_array_exp)
    cam_resized = cv2.resize(cam, TARGET_SIZE)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(cam_uint8, COLORMAP)
    overlay_bgr = cv2.addWeighted(resized_bgr, 1.0 - ALPHA, heatmap_colored, ALPHA, 0)

    hotspots, heatmask = detect_hotspots_from_heatmap(
        cam_uint8,
        threshold_ratio=threshold_ratio,
        min_area=min_area,
        lung_mask=lung_mask
    )

    marked_overlay = draw_hotspots_on_image(hotspots, overlay_bgr, text=annot_text)

    mask_filled = np.zeros_like(resized_bgr, dtype=np.uint8)
    for (center, radius, _) in hotspots:
        cv2.circle(mask_filled, center, radius, CIRCLE_COLOR, -1)
    filled_overlay = cv2.addWeighted(resized_bgr, 0.6, mask_filled, 0.4, 0)
    filled_with_numbers = draw_hotspots_on_image(hotspots, filled_overlay.copy(), text=annot_text)

    base = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = os.path.join(IMAGE_SAVE_DIR, f"{base}_{timestamp}")

    orig_path = out_prefix + "_orig.png"
    mask_path = out_prefix + "_mask.png"
    overlay_path = out_prefix + "_overlay.png"
    heatmap_path = out_prefix + "_heatmap.png"
    filled_path = out_prefix + "_filled.png"

    cv2.imwrite(orig_path, resized_bgr)
    cv2.imwrite(mask_path, cv2.cvtColor(lung_mask, cv2.COLOR_GRAY2BGR) if lung_mask is not None else np.zeros_like(resized_bgr))
    cv2.imwrite(overlay_path, marked_overlay)
    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(filled_path, filled_with_numbers)

    df_stats = hotspot_statistics(hotspots, cam_uint8)
    severity_score = compute_severity_score(df_stats)

    summary_row = {
        "image": base,
        "timestamp": timestamp,
        "prediction": pred,
        "label": label,
        "confidence": confidence,
        "n_hotspots": len(hotspots),
        "severity_score": severity_score
    }

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
        "summary_row": summary_row
    }

# -------------------------
# Batch processing helper
# -------------------------
def process_batch_images(image_paths, threshold_ratio=THRESHOLD_RATIO_DEFAULT, min_area=MIN_AREA_DEFAULT, use_lung_mask=True):
    all_summaries = []
    detailed_stats = {}
    for p in image_paths:
        print(f"Processing {p} ...")
        try:
            out = process_single_image(p, threshold_ratio, min_area, use_lung_mask)
            all_summaries.append(out["summary_row"])
            detailed_stats[p] = out
            print(f"Saved outputs for {Path(p).stem}")
        except Exception as e:
            print(f"Failed {p}: {e}")
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(IMAGE_SAVE_DIR, "batch_summary.csv"), index=False)
    return df_summary, detailed_stats

# -------------------------
# Interactive UI (Colab friendly)
# -------------------------
upload_btn = widgets.FileUpload(accept=".png,.jpg,.jpeg,.zip", multiple=True)
threshold_slider = widgets.FloatSlider(value=THRESHOLD_RATIO_DEFAULT, min=0.05, max=0.99, step=0.01, description='Threshold')
minarea_slider = widgets.IntSlider(value=MIN_AREA_DEFAULT, min=1, max=1000, step=1, description='MinArea')
lung_mask_toggle = widgets.Checkbox(value=True, description='Use Lung Mask')
process_button = widgets.Button(description="Process Uploads", button_style='primary')
out_box = widgets.Output(layout={'border': '1px solid black'})

def handle_process(b):
    with out_box:
        clear_output()
        if not upload_btn.value:
            print("Please upload one or more image files (or a zip).")
            return

        saved_paths = []
        for fn, fileinfo in upload_btn.value.items():
            content = fileinfo['content']
            if fn.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    z.extractall("uploaded_zip")
                for root, _, files in os.walk("uploaded_zip"):
                    for f in files:
                        if f.lower().endswith((".png",".jpg",".jpeg")):
                            saved_paths.append(os.path.join(root, f))
            else:
                outpath = os.path.join("uploaded_images", fn)
                with open(outpath, "wb") as f:
                    f.write(content)
                saved_paths.append(outpath)

        if not saved_paths:
            print("No image files found in uploads.")
            return

        print(f"Found {len(saved_paths)} images. Starting processing...")
        df_summary, _ = process_batch_images(
            saved_paths,
            threshold_ratio=threshold_slider.value,
            min_area=minarea_slider.value,
            use_lung_mask=lung_mask_toggle.value
        )

        print("\nBatch Summary:")
        display(df_summary)

        zip_out = os.path.join(IMAGE_SAVE_DIR, "all_results.zip")
        with zipfile.ZipFile(zip_out, 'w') as zf:
            for root, _, files in os.walk(IMAGE_SAVE_DIR):
                for f in files:
                    if not f.endswith('.zip'):
                        zf.write(os.path.join(root, f), os.path.join(os.path.basename(root), f))

        print(f"\nAll outputs zipped: {zip_out}")
        from google.colab import files
        files.download(zip_out)

process_button.on_click(handle_process)

ui = widgets.VBox([
    widgets.HTML("<h3>Futuristic Grad-CAM + Hotspot Analytics Pipeline</h3>"),
    widgets.HBox([upload_btn, process_button]),
    widgets.HBox([threshold_slider, minarea_slider, lung_mask_toggle]),
    out_box
])
display(ui)

print("UI ready — upload images (or a ZIP) and click 'Process Uploads'.")
