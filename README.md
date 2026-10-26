# ReBorn AI: Agentic Waste Vision on AWS

ReBorn AI inspects a photo of a waste item with **OpenCV 5**, looks again when it is unsure, and decides: **Reuse**, **Reprocess**, or **Dispose**. When the item is uncertain or possibly hazardous, it stops and asks a human before advising. Results are explained in **English and Tamil**.

Built for the **OpenCV AI Competition 2026, powered by AWS** (featured path: **Agentic Vision**).

| | |
|---|---|
| **Live demo** | [http://YOUR-EC2-PUBLIC-IP:5000] |
| **Demo video** | [YouTube unlisted link] |
| **Technical report** | [link to report PDF in this repo, or Devpost] |
| **Code** | this repository |

> If the live endpoint is down when you read this, the team can give a live screen-share demo. Contact: [email].

## The problem

People often can't tell what can be reused, recycled, or disposed of safely. Mixed waste goes to landfill, and hazardous items in normal garbage put waste collectors at risk. Users: students, households, shops, and waste collectors.

## What it does

From one photo, the agent runs a perception, decision, and action loop:

1. **Checks image quality** (blur, brightness, glare). A fatal problem means "retake the photo". A fixable problem triggers enhancement.
2. **Segments the item** and puts it on a white background.
3. **Classifies two views** (full photo and cutout) with an ONNX model through OpenCV's DNN module.
4. **Re-inspects when unsure.** If the views disagree or confidence is below 0.70, it classifies zoomed and flipped views and compares.
5. **Stops and asks a human** if it is still unsure, or if a hazard rule fires.
6. **Analyses condition** (edge density and a rust or stain colour check).
7. **Decides Reuse / Reprocess / Dispose** and explains in English and Tamil.

Every step is written to a **trace**: *OpenCV output, then decision, then next action*. The web page shows it, and a JSON copy is saved per image.

### Why this is agentic

The visual evidence changes what the system does next:

| OpenCV output | Next action |
|---|---|
| Blur or brightness below the threshold | Enhance the image, or ask for a retake |
| Full-photo and cutout labels disagree, or confidence is low | Zoom and re-classify |
| Still unsure after the zoom, or the hazard rule fires | Stop and wait for human confirmation |
| Condition score above or below 0.65 | Choose Reuse or Reprocess |

A sample trace is in [docs/sample_trace.txt](docs/sample_trace.txt).

## Architecture

![Architecture](docs/architecture.png)
![Agent workflow](docs/agent_workflow.png)

- **Web app:** Flask, served by gunicorn on an AWS **EC2** instance ([instance type], [region]).
- **Vision:** OpenCV 5 (`opencv-python-headless`), with an ONNX MobileNetV2 classifier loaded via `cv2.dnn`.
- **Storage:** [Amazon S3 bucket stores each uploaded image and result JSON. / Not used, results are stored on the instance disk.]
- **Access:** [IAM role attached to the instance, with no keys in code.]
- **Explanations:** fixed English and Tamil templates in `agent.py` (no LLM is used).

## OpenCV 5 functions used

| Step | Functions |
|---|---|
| Quality check | `cv2.cvtColor`, `cv2.resize` (`INTER_AREA`), `cv2.Laplacian` |
| Enhancement | `cv2.createCLAHE` (on the LAB L channel), `cv2.GaussianBlur`, `cv2.addWeighted` |
| Segmentation | `cv2.grabCut`, `cv2.morphologyEx`, `cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect` |
| Classification | `cv2.dnn.readNetFromONNX`, `cv2.dnn.blobFromImage`, `net.forward()` |
| Re-check views | `cv2.flip`, array cropping and re-classification |
| Condition | `cv2.Canny`, `cv2.erode`, HSV conversion with a colour mask |
| Output | `cv2.rectangle`, `cv2.putText`, `cv2.imencode`/`imwrite` |

## Model

- MobileNetV2 fine-tuned on **TrashNet** (6 classes: cardboard, glass, metal, paper, plastic, trash), exported to ONNX.
- Training used Google Colab. Validation accuracy on held-out TrashNet images: **[XX%]**.
- The weights are in `model/waste.onnx` and the class names in `model/labels.txt`.

## Run locally (Windows, macOS, Linux)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__)"   # should print 5.x
python app.py
```

Open http://localhost:5000, upload a photo, and read the result and trace.

Command-line test (prints the trace):

```bash
python agent.py bottle.jpg
python test_model.py bottle.jpg      # raw classifier output only
```

## Deploy on AWS (EC2)

1. Launch an **Ubuntu 24.04** EC2 instance. Open **port 5000** in the security group (and SSH from your IP only).
2. Optional: create an S3 bucket and attach an IAM role that allows `s3:PutObject` and `s3:GetObject` on that bucket only.
3. On the instance:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git libglib2.0-0
git clone https://github.com/kit25cseasherlins-dotcom/reborn-ai-agentic-waste-vision.git
cd reborn-ai-agentic-waste-vision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

4. Run with gunicorn using **one worker** (results are kept in memory):

```bash
gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:5000 app:app
```

For a permanent service, use a systemd unit with the same `ExecStart` command. Set `REBORN_BUCKET` and `AWS_DEFAULT_REGION` as environment variables if you use S3.

## Evaluation

| Test | Images | Result |
|---|---|---|
| Classifier on held-out TrashNet images | [N] | [XX%] accuracy |
| Team's own phone photos (full agent) | [N] | [XX%] correct material |
| Own photos sent to human review | [N] | [XX%] |
| Own photos wrongly accepted without review | [N] | [XX%] |
| Average processing time per image on [instance] | [N] | [X.X] s |

Evaluation scripts and raw outputs: [docs/evaluation.md].

### Failure cases (real examples)

- [Example 1: e.g. clear plastic bottle on a brown table predicted as cardboard, confidence 0.64, sent to human review.]
- [Example 2: describe a case that was wrongly accepted, if any.]
- [Example 3.]

**Known weakness:** the classifier was trained on clean single-item photos with plain backgrounds. Cluttered or brown backgrounds push predictions towards "cardboard". The cutout and re-check steps reduce this, and the human step catches many remaining errors, but not all.

## Limitations

- Only 6 material classes. Items outside them (food, e-waste, medical waste) are not recognised.
- Accuracy drops on dirty, transparent, crushed, or unusual items and on cluttered backgrounds.
- The hazard check is a **shape heuristic only** (small cylindrical item). It is **not** a hazardous-material detector.
- Condition analysis (edges and colour) is a rough estimate, not an inspection.
- Reuse, Reprocess, and Dispose advice is generic and does not replace local waste rules.
- Tamil text uses fixed templates and should be reviewed by a native speaker.
- Results are held in memory, so they are lost if the server restarts (saved copies are in `outputs/` and S3, if enabled).

## Responsible use

- No personal data is collected, and no login is needed.
- Uploaded images and results are stored for [retention period / "for the competition judging period"], then deleted.
- The system asks a human to confirm uncertain or possibly hazardous items instead of guessing.
- Do not use it as the only basis for handling dangerous waste.
- Security: no keys are stored in code, `debug` is off in production, and the instance role is limited to one bucket.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Flask web app (upload page, trace view, human confirmation) |
| `agent.py` | Agent loop, decision policy, trace, English and Tamil text |
| `tools.py` | OpenCV tools (quality, enhance, segment, classify, re-check, condition, hazard rule) |
| `test_model.py` | Raw classifier test |
| `model/` | `waste.onnx` and `labels.txt` |
| `requirements.txt` | Pinned dependencies |
| `docs/` | Diagrams, sample trace, evaluation notes |

## Future work

Hazardous-item classes (batteries, e-waste), live video, offline on-device mode, Tamil voice, a real reuse marketplace, training on photos with real-world backgrounds, and recycler partnerships.

## Team

[Name, role] · [Name, role] · [Name, role] · [Name, role]

## Credits

- OpenCV 5, Flask, gunicorn, NumPy, PyTorch (training only).
- TrashNet dataset (Yang and Thung, Stanford).
- Hosted on AWS.

## License

MIT (see [LICENSE](LICENSE)).
