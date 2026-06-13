import json
import os
from collections import defaultdict
from tqdm import tqdm
import sys

# --- 1. Configure paths ---
JSON_FILE = '/home/sysspace/yanght/test/DPDETR/DPDETR-main/dataset/rbox_Drone_Vehicle/val_ir_segmentation.json'
OUTPUT_DIR = '/home/sysspace/yanght/test/DPDETR/DPDETR-main/dataset/val/ir/label'
# --------------------

def convert_coco_obb_to_yolo():
    """
    Convert COCO OBB JSON format to YOLO OBB .txt format.
    - The "segmentation" field is assumed to be [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    print(f"--- COCO OBB to YOLO OBB ---")
    print(f"Loading JSON file: {JSON_FILE}...")

    # --- 2. Load JSON file ---
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {JSON_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON. Please check file content.")
        sys.exit(1)

    # --- 3. Create lookup table (for efficient retrieval) ---
    print("Creating image ID -> info (width, height, filename) mapping...")
    image_id_to_info = {}
    for image in data['images']:
        image_id_to_info[image['id']] = {
            'width': image['width'],
            'height': image['height'],
            'file_name': image['file_name']
        }

    print("Grouping annotations by image ID...")
    annotations_by_image = defaultdict(list)

    # Filter valid OBB annotations
    for ann in data['annotations']:
        seg = ann.get('segmentation')
        if not seg:
            continue

        seg_points = []
        # COCO segmentation format may be [[x1,y1...]]
        if isinstance(seg, list) and len(seg) == 1 and isinstance(seg[0], list):
             seg_points = seg[0]
        # Or flattened format [x1,y1...]
        elif isinstance(seg, list) and len(seg) > 1:
             seg_points = seg
        else:
             continue  # Skip invalid format

        # Only process 8-point OBB annotations
        if len(seg_points) == 8:
            ann['segmentation_points'] = seg_points  # Store processed 8 points
            annotations_by_image[ann['image_id']].append(ann)

    # --- 4. Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory confirmed: {OUTPUT_DIR}")

    # --- 5. Convert and write files in a loop ---
    print(f"Converting annotations for {len(image_id_to_info)} images...")
    converted_count = 0
    skipped_count = 0

    # Use tqdm for progress bar
    for image_id, info in tqdm(image_id_to_info.items()):
        img_w = info['width']
        img_h = info['height']
        file_name = info['file_name']

        # Robustness check: prevent division by zero
        if img_w == 0 or img_h == 0:
            print(f"Warning: Image {file_name} (ID: {image_id}) has width or height 0, skipped.")
            skipped_count += 1
            continue

        # Get all annotations for this image
        annotations_list = annotations_by_image.get(image_id, [])

        yolo_lines = []

        for ann in annotations_list:
            category_id = ann['category_id']
            category_id = category_id - 1

            # Use our preprocessed 8 points
            points = ann['segmentation_points']

            try:
                # [x1, y1, x2, y2, x3, y3, x4, y4]
                x1, y1, x2, y2, x3, y3, x4, y4 = points

                # --- Normalization ---
                x1_norm = x1 / img_w
                y1_norm = y1 / img_h
                x2_norm = x2 / img_w
                y2_norm = y2 / img_h
                x3_norm = x3 / img_w
                y3_norm = y3 / img_h
                x4_norm = x4 / img_w
                y4_norm = y4 / img_h

                # Format: class x1 y1 x2 y2 x3 y3 x4 y4 (8 decimal places)
                line = (f"{category_id} {x1_norm:.8f} {y1_norm:.8f} {x2_norm:.8f} {y2_norm:.8f} "
                        f"{x3_norm:.8f} {y3_norm:.8f} {x4_norm:.8f} {y4_norm:.8f}")
                yolo_lines.append(line)

            except Exception as e:
                print(f"Error: processing annotation {ann['id']} (image: {file_name}): {e}")

        # --- 6. Write .txt file ---
        # Generate "00001.txt" from "00001.jpg"
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        try:
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")  # Add trailing newline if file is non-empty
            converted_count += 1
        except Exception as e:
            print(f"Error: writing file {txt_path}: {e}")
            skipped_count += 1

    print("\n--- Conversion complete ---")
    print(f"Successfully converted and saved: {converted_count} files.")
    print(f"Skipped (due to errors or image size 0): {skipped_count} files.")

if __name__ == "__main__":
    convert_coco_obb_to_yolo()
