import json
import sys
import cv2
import numpy as np
import tools as T

LABELS = T.LABELS
REUSE_OK = {"glass", "metal", "plastic", "cardboard"}
TA = {"cardboard": "அட்டை", "glass": "கண்ணாடி", "metal": "உலோகம்",
      "paper": "காகிதம்", "plastic": "பிளாஸ்டிக்", "trash": "குப்பை"}
BIN = {"cardboard": "dry recycling (flatten it, keep it dry)",
       "glass": "glass recycling (wrap it if broken)",
       "metal": "metal recycling", "paper": "paper recycling",
       "plastic": "plastic recycling (rinse it first)",
       "trash": "general waste bin"}


def _step(trace, tool, observation, decision, nxt):
    trace.append({"step": len(trace) + 1, "tool": tool, "opencv_output": observation,
                  "decision": decision, "next_action": nxt})


def decide(label, cond):
    """Policy: choose Reuse / Reprocess / Dispose."""
    good = cond["score"] >= 0.65
    if label == "trash":
        return "DISPOSE"
    if label in REUSE_OK and good:
        return "REUSE"
    return "REPROCESS"


def texts(label, decision, cond, hazard=""):
    en = {"REUSE": f"This {label} item looks in good condition. It can be reused or listed for someone who needs it.",
          "REPROCESS": f"This {label} item can be recycled or upcycled. Take it to a recycler or the {BIN[label]}.",
          "DISPOSE": "This item cannot be reused or recycled. Put it in the general waste bin."}[decision]
    ta = {"REUSE": f"இந்த {TA[label]} பொருள் நல்ல நிலையில் உள்ளது. மீண்டும் பயன்படுத்தலாம் அல்லது தேவைப்படுபவர்களுக்கு கொடுக்கலாம்.",
          "REPROCESS": f"இந்த {TA[label]} பொருளை மறுசுழற்சி செய்யலாம். மறுசுழற்சி மையத்திலோ சரியான தொட்டியிலோ போடவும்.",
          "DISPOSE": "இந்த பொருளை மறுபயன்பாடு அல்லது மறுசுழற்சி செய்ய முடியாது. பொது குப்பைத் தொட்டியில் போடவும்."}[decision]
    if hazard:
        en += " SAFETY: a human has checked this item; keep it away from normal garbage if it is a battery or chemical."
        ta += " பாதுகாப்பு: இது பேட்டரி அல்லது ரசாயனமாக இருந்தால் சாதாரண குப்பையுடன் சேர்க்க வேண்டாம்."
    if label == "glass" and cond["text"] != "good":
        en += " Wrap damaged glass before disposal."
        ta += " உடைந்த கண்ணாடியை சுற்றி வைத்து பாதுகாப்பாக போடவும்."
    return en, ta


def finish(state, label, human=False):
    """Final decision once the material is known (by the AI or by a human)."""
    cond = state["condition"]
    decision = decide(label, cond)
    en, ta = texts(label, decision, cond, state.get("hazard", "") if human else "")
    _step(state["trace"], "decide",
          {"material": label, "condition_score": cond["score"], "confirmed_by_human": human},
          f"policy -> {decision}", "explain in English and Tamil")
    state.update({"status": "done", "label": label, "decision": decision,
                  "en": en, "ta": ta})
    return state


def run(img_bgr):
    trace = []
    state = {"trace": trace, "status": "done"}
    img = T.limit_size(img_bgr, 800)

    # 1) Quality gate
    q = T.check_quality(img)
    if q["fatal"]:
        _step(trace, "check_quality", q, "unusable image", "ask user to retake")
        state.update({"status": "retake", "label": None, "decision": "RETAKE",
                      "en": "The photo is too blurry or dark. Please retake it in good light.",
                      "ta": "புகைப்படம் தெளிவாக இல்லை. நல்ல வெளிச்சத்தில் மீண்டும் எடுக்கவும்."})
        return state, img
    if q["issues"]:
        _step(trace, "check_quality", q, "image poor: " + ", ".join(q["issues"]),
              "enhance (CLAHE + sharpen)")
        img = T.enhance(img)
        q2 = T.check_quality(img)
        _step(trace, "enhance", {"brightness": q2["brightness"], "glare": q2["glare"]},
              "enhanced image used", "segment item")
    else:
        _step(trace, "check_quality", q, "image quality OK", "segment item")

    # 2) Segment the item
    seg = T.segment_item(img)
    _step(trace, "segment_item",
          {"ok": seg["ok"], "area_ratio": seg["area_ratio"], "bbox": [int(v) for v in seg["bbox"]]},
          "item cut out on white background" if seg["ok"] else "segmentation weak: used centre crop",
          "classify full photo and cutout")

    # 3) Classify twice and compare
    p_full = T.classify(img)
    p_crop = T.classify(seg["crop"])
    lf, lc = LABELS[int(p_full.argmax())], LABELS[int(p_crop.argmax())]
    cf, cc = float(p_full.max()), float(p_crop.max())
    label, conf, review = lc, cc, ""
    if lf == lc and cc >= 0.70:
        _step(trace, "classify", {"full": f"{lf} {cf:.2f}", "cutout": f"{lc} {cc:.2f}"},
              "both views agree", "analyse condition")
    else:
        _step(trace, "classify", {"full": f"{lf} {cf:.2f}", "cutout": f"{lc} {cc:.2f}"},
              "views disagree or confidence low", "zoom and re-check")
        p_z = T.zoom_recheck(seg["crop"])
        lz, cz = LABELS[int(p_z.argmax())], float(p_z.max())
        if cz >= 0.75 and lz == lc:
            label, conf = lz, cz
            _step(trace, "zoom_recheck", {"result": f"{lz} {cz:.2f}"},
                  "zoom confirms cutout result", "analyse condition")
        else:
            label, conf = lz, cz
            review = f"AI unsure (full: {lf} {cf:.2f}, cutout: {lc} {cc:.2f}, zoom: {lz} {cz:.2f})"
            _step(trace, "zoom_recheck", {"result": f"{lz} {cz:.2f}"},
                  "still unsure", "STOP: ask a human to confirm the material")

    # 4) Condition + hazard heuristic
    cond = T.analyze_condition(img, seg, label)
    _step(trace, "analyze_condition", {k: cond[k] for k in ("score", "edge_density", "stain_ratio")},
          f"condition: {cond['text']}", "check hazard rule")
    hazard = T.possible_hazard(label, seg)

    state.update({"label": label, "confidence": round(conf, 2), "condition": cond,
                  "hazard": hazard})

    # annotated image
    ann = img.copy()
    x, y, bw, bh = seg["bbox"]
    cv2.rectangle(ann, (x, y), (x + bw, y + bh), (0, 200, 0), 3)
    cv2.putText(ann, f"{label} {conf:.2f}", (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    if review or hazard:
        reason = hazard or review
        _step(trace, "human_gate", {"reason": reason}, "human approval required",
              "wait for human to confirm material")
        state.update({"status": "needs_review", "decision": "WAITING FOR HUMAN",
                      "review_reason": reason,
                      "en": "Not sure or possibly hazardous. A human must confirm the material first.",
                      "ta": "உறுதியாக தெரியவில்லை அல்லது ஆபத்தாக இருக்கலாம். முதலில் மனிதர் பொருளை உறுதிசெய்ய வேண்டும்."})
        return state, ann

    _step(trace, "human_gate", {"reason": "none"}, "no approval needed", "decide")
    return finish(state, label), ann


if __name__ == "__main__":
    for path in sys.argv[1:]:
        im = cv2.imread(path)
        if im is None:
            print(path, "-> cannot read")
            continue
        st, _ = run(im)
        print("=" * 60, "\n", path, "->", st.get("label"), st.get("decision"), "| status:", st["status"])
        for s in st["trace"]:
            print(f"  {s['step']}. {s['tool']}: {s['opencv_output']} => {s['decision']} -> {s['next_action']}")
        print("  EN:", st["en"])
        print("  TA:", st["ta"])