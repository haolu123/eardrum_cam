from collections import defaultdict
import os

def get_valid_classes(root_dir, min_images=5000):
    class_counts = defaultdict(int)
    for root, _, files in os.walk(root_dir):
        class_name = os.path.basename(root)
        if class_name not in ['.', '..']:
            class_counts[class_name] += len([f for f in files if f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')])
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_images]
    print(f"Classes with more than {min_images} images: {valid_classes}")
    return valid_classes, class_counts