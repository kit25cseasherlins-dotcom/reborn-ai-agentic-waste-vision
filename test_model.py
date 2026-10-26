import cv2, numpy as np, sys

print("OpenCV", cv2.__version__)
net = cv2.dnn.readNetFromONNX("model/waste.onnx")
labels = open("model/labels.txt").read().strip().split("\n")

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

for path in sys.argv[1:]:
    img = cv2.imread(path)
    if img is None:
        print(path, "-> could not read image")
        continue
    small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(small, 1/255.0, (224, 224), swapRB=True)
    net.setInput((blob - mean) / std)
    out = net.forward()[0]
    probs = np.exp(out - out.max()); probs /= probs.sum()
    top = probs.argsort()[::-1][:3]
    print(path, "->", ", ".join(f"{labels[i]} {probs[i]:.2f}" for i in top))