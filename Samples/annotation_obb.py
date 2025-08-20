import os
import cv2
import numpy as np

def parse_dota_annotation(txt_file):
    """
    Parse a DOTA annotation file.
    Each line (after the header) contains:
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    """
    boxes = []
    labels = []
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:  # invalid line
                continue
            coords = list(map(float, parts[:8]))
            label = parts[8] if len(parts) > 8 else "unknown"
            boxes.append(np.array(coords).reshape(4, 2))  # 4 points (x,y)
            labels.append(label)
    return boxes, labels

def draw_dota_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        pts = box.astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Put label at the first point
        cv2.putText(image, label, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

def annotate_dota_images(img_dir, ann_dir, save_dir="annotated"):
    os.makedirs(save_dir, exist_ok=True)

    for txt_file in os.listdir(ann_dir):
        if not txt_file.endswith(".txt"):
            continue
        base_name = os.path.splitext(txt_file)[0]
        img_path = os.path.join(img_dir, base_name + ".jpg")  # change if .jpg/.tif
        ann_path = os.path.join(ann_dir, txt_file)

        if not os.path.exists(img_path):
            print(f"Image not found for {txt_file}, skipping.")
            continue

        image = cv2.imread(img_path)
        boxes, labels = parse_dota_annotation(ann_path)
        annotated = draw_dota_boxes(image, boxes, labels)

        out_path = os.path.join(save_dir, base_name + "_annotated.png")
        cv2.imwrite(out_path, annotated)
        print(f"Saved {out_path}")

# Example usage:
for split in ["train","val","test"]:
    imgs=os.path.join("DRASTI_Samples",split,"images")
    lbls=os.path.join("DRASTI_Samples",split,"labels")
    out=os.path.join("DRASTI_Samples",split,"annotated")
    annotate_dota_images(imgs,lbls,out)
