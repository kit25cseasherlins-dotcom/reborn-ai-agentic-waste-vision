import os
import uuid
import json
import traceback
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory, redirect
import agent

app = Flask(__name__)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = {}
BUCKET = os.environ.get("REBORN_BUCKET")   # optional: S3 bucket name


def save_json(rid, state):
    path = os.path.join(OUT, rid + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str, ensure_ascii=False)
    return path


def save_s3(path, key):
    if not BUCKET:
        return
    try:
        import boto3
        boto3.client("s3").upload_file(path, BUCKET, key)
    except Exception as e:
        print("S3 upload skipped:", e)


PAGE = """
<!doctype html><meta charset="utf-8"><meta name=viewport content="width=device-width,initial-scale=1">
<title>ReBorn AI</title>
<style>
body{font-family:system-ui,sans-serif;max-width:820px;margin:20px auto;padding:0 14px;color:#1b1b1b}
.card{border:1px solid #ddd;border-radius:12px;padding:14px;margin:12px 0}
.done{background:#eaf7ee}.needs_review{background:#fff4e0}.retake{background:#fdeaea}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}
img{max-width:100%;border-radius:8px}button{padding:10px 16px;font-size:16px}
pre{white-space:pre-wrap;background:#222;color:#eee;padding:10px;border-radius:8px;font-size:12px}
</style>
<h1>ReBorn AI</h1>
<p>Upload a photo of a waste item. The agent inspects it with OpenCV 5 and decides: Reuse, Reprocess or Dispose.</p>
<form method=post action=/analyze enctype=multipart/form-data class=card>
<input type=file name=photo accept="image/*" required>
<button>Analyse</button></form>
{% if err %}<div class=card><h3>Error (copy this and send it)</h3><pre>{{ err }}</pre></div>{% endif %}
{% if r %}
<div class="card {{ r['status'] }}">
<h2>{{ r['decision'] }}{% if r['label'] %} &mdash; {{ r['label'] }}{% endif %}</h2>
{% if r['rid'] %}<img src="/files/{{ r['rid'] }}.jpg">{% endif %}
<p><b>English:</b> {{ r['en'] }}</p>
<p><b>தமிழ்:</b> {{ r['ta'] }}</p>
{% if r.get('condition') %}<p>Condition: {{ r['condition']['text'] }} (score {{ r['condition']['score'] }})</p>{% endif %}
{% if r['status'] == 'needs_review' %}
<form method=post action="/approve/{{ r['rid'] }}">
<p><b>Human check:</b> {{ r.get('review_reason') }}<br>Confirm the material:
<select name=label>{% for l in labels %}<option>{{ l }}</option>{% endfor %}</select>
<button>Confirm</button></p></form>
{% endif %}
</div>
<div class=card><h3>Agent trace (OpenCV output &rarr; decision &rarr; next action)</h3>
<table><tr><th>#</th><th>Tool</th><th>OpenCV output</th><th>Decision</th><th>Next</th></tr>
{% for s in r['trace'] %}<tr><td>{{ s['step'] }}</td><td>{{ s['tool'] }}</td><td>{{ s['opencv_output'] }}</td>
<td>{{ s['decision'] }}</td><td>{{ s['next_action'] }}</td></tr>{% endfor %}</table></div>
{% endif %}
"""


def page(r=None, err=None):
    return render_template_string(PAGE, r=r, err=err, labels=agent.LABELS)


@app.route("/")
def home():
    return page()


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = np.frombuffer(request.files["photo"].read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return page(err="Could not read that image. Use a JPG or PNG."), 400
        state, ann = agent.run(img)
        rid = uuid.uuid4().hex[:10]
        state["rid"] = rid
        cv2.imwrite(os.path.join(OUT, rid + ".jpg"), ann)
        RESULTS[rid] = state
        jpath = save_json(rid, state)
        save_s3(os.path.join(OUT, rid + ".jpg"), f"images/{rid}.jpg")
        save_s3(jpath, f"results/{rid}.json")
        return page(r=state)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return page(err=tb), 500


@app.route("/approve/<rid>", methods=["POST"])
def approve(rid):
    try:
        if rid not in RESULTS:
            return redirect("/")
        state = RESULTS[rid]
        agent.finish(state, request.form["label"], human=True)
        jpath = save_json(rid, state)
        save_s3(jpath, f"results/{rid}.json")
        return page(r=state)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return page(err=tb), 500


@app.route("/files/<name>")
def files(name):
    return send_from_directory(OUT, name)


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)