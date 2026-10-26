import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
NET = cv2.dnn.readNetFromONNX(os.path.join(BASE, "model", "waste.onnx"))
LABELS = open(os.path.join(BASE, "model", "labels.txt")).read().strip().split("\n")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def limit_size(img, max_side=800):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ---------- Tool 1: image quality ----------
def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (640, max(1, int(640 * gray.shape[0] / gray.shape[1]))),
                   interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    brightness = float(gray.mean())
    glare = float((gray > 245).mean())
    issues = []
    if blur < 50:
        issues.append("blurry")
    if brightness < 60:
        issues.append("too dark")
    if brightness > 210:
        issues.append("too bright")
    if glare > 0.12:
        issues.append("glare")
    fatal = blur < 25 or brightness < 35
    return {"blur": round(blur, 1), "brightness": round(brightness, 1),
            "glare": round(glare, 3), "issues": issues, "fatal": fatal}


# ---------- Tool 2: enhancement ----------
def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    soft = cv2.GaussianBlur(out, (0, 0), 2)
    return cv2.addWeighted(out, 1.5, soft, -0.5, 0)


# ---------- Tool 3: segment the item (GrabCut) ----------
def _pad_square_white(img):
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def segment_item(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    ok = True
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise ValueError("no contour")
        c = max(cnts, key=cv2.contourArea)
        clean = np.zeros_like(fg)
        cv2.drawContours(clean, [c], -1, 255, -1)
        ratio = float(cv2.contourArea(c)) / (h * w)
        if ratio < 0.03 or ratio > 0.92:
            raise ValueError("bad area ratio %.2f" % ratio)
        x, y, bw, bh = cv2.boundingRect(c)
    except Exception:
        ok = False
        bw, bh = int(w * 0.7), int(h * 0.7)
        x, y = (w - bw) // 2, (h - bh) // 2
        clean = np.zeros((h, w), np.uint8)
        clean[y:y + bh, x:x + bw] = 255
        ratio = 0.49

    white = np.full_like(img, 255)
    white[clean > 0] = img[clean > 0]
    px, py = int(bw * 0.08), int(bh * 0.08)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    crop = _pad_square_white(white[y0:y1, x0:x1])
    return {"ok": ok, "mask": clean, "bbox": (x, y, bw, bh), "crop": crop,
            "area_ratio": round(ratio, 3), "aspect": round(bw / max(bh, 1), 2)}


# ---------- Tool 4: classify (OpenCV DNN + ONNX) ----------
def classify(img):
    small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(small, 1 / 255.0, (224, 224), swapRB=True)
    NET.setInput((blob - MEAN) / STD)
    out = NET.forward()[0]
    p = np.exp(out - out.max())
    return p / p.sum()


# ---------- Tool 5: zoom and re-check (test-time views) ----------
def zoom_recheck(crop):
    h, w = crop.shape[:2]
    views = [crop, cv2.flip(crop, 1)]
    for f in (0.85, 0.7):
        dh, dw = int(h * (1 - f) / 2), int(w * (1 - f) / 2)
        views.append(crop[dh:h - dh, dw:w - dw])
    return np.mean([classify(v) for v in views], axis=0)


# ---------- Tool 6: condition analysis ----------
def analyze_condition(img, seg, label):
    mask = seg["mask"]
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))
    n = int((inner > 0).sum())
    if n < 200:
        return {"score": 0.5, "edge_density": 0.0, "stain_ratio": 0.0,
                "text": "unknown", "note": "item region too small"}
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float((edges[inner > 0] > 0).mean())
    stain = 0.0
    if label in ("metal", "plastic", "glass"):   # rust/stain colour check
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        rust = ((h >= 5) & (h <= 22) & (s > 110) & (v > 50) & (v < 200)) & (inner > 0)
        stain = float(rust.sum()) / n
    damage = min(1.0, edge_density / 0.15) * 0.5 + min(1.0, stain / 0.30) * 0.5
    score = round(1.0 - damage, 2)
    text = "good" if score >= 0.65 else "worn or damaged"
    return {"score": score, "edge_density": round(edge_density, 3),
            "stain_ratio": round(stain, 3), "text": text, "note": ""}


# ---------- Tool 7: simple hazard heuristic ----------
def possible_hazard(label, seg):
    small = seg["area_ratio"] < 0.35
    cylinder = seg["aspect"] >= 1.7 or seg["aspect"] <= 0.6
    if label in ("metal", "trash") and small and cylinder:
        return "small cylindrical item (could be a battery or aerosol can)"
    return ""