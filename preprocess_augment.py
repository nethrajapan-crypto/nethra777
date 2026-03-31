"""
STEP 3: Preprocess & Augment Images (128x128)
Loads images from source folders, resizes to 128x128, applies augmentation,
and organizes them into train/val/test splits.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# ==================== Configuration ====================
SOURCE_DEFECT_CROPS = './defect_crops'  # Defect images
SOURCE_IMAGES = './images'               # No defect images (if available)
DATASET_ROOT = './dataset'
IMAGE_SIZE = 128

# ==================== Augmentation Pipeline ====================
def apply_augmentation(image):
    """
    Apply random augmentations to image (OpenCV-based)
    """
    h, w = image.shape[:2]
    
    # Random horizontal flip (50%)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random vertical flip (50%)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
    
    # Random rotation (±15 degrees, 50%)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness adjustment (20%)
    if random.random() > 0.8:
        brightness = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Random contrast adjustment (20%)
    if random.random() > 0.8:
        contrast = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    
    # Random Gaussian noise (20%)
    if random.random() > 0.8:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    
    return image

# ==================== Load and Split Images ====================
def load_images_from_folder(folder_path, label_name):
    """
    Load all images from a folder and return with labels
    """
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist")
        return images, labels
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(os.path.join(root, file))
                labels.append(label_name)
    
    return images, labels

def preprocess_image(image_path, output_path, augment=False):
    """
    Load, resize, and optionally augment a single image
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 128x128
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        # Apply augmentation for training images
        if augment:
            image = apply_augmentation(image)
        
        # Convert back to BGR for saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save as PNG
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

# ==================== Main Processing ====================
def main():
    print("=" * 60)
    print("STEP 3: Preprocessing & Augmenting Images (128x128)")
    print("=" * 60)
    
    # Step 1: Collect all images
    print("\n[1/4] Collecting images from source folders...")
    all_images = []
    all_labels = []
    
    # Load defect images
    defect_images, defect_labels = load_images_from_folder(
        SOURCE_DEFECT_CROPS, 'defect'
    )
    all_images.extend(defect_images)
    all_labels.extend(defect_labels)
    print(f"  ✓ Found {len(defect_images)} defect images")
    
    # Load no-defect images (if source_images exists)
    if os.path.exists(SOURCE_IMAGES):
        no_defect_images, no_defect_labels = load_images_from_folder(
            SOURCE_IMAGES, 'no_defect'
        )
        all_images.extend(no_defect_images)
        all_labels.extend(no_defect_labels)
        print(f"  ✓ Found {len(no_defect_images)} no-defect images")
    else:
        print(f"  ⚠ No no-defect source folder found. Add images to {SOURCE_IMAGES}")
    
    if not all_images:
        print("\n❌ No images found! Check source folders and try again.")
        return
    
    print(f"  ✓ Total images collected: {len(all_images)}")
    
    # Step 2: Split into train/val/test (70/15/15)
    print("\n[2/4] Splitting dataset (70% train, 15% val, 15% test)...")
    
    # First split: 70% train, 30% temp
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    # Second split: 50/50 of remaining for val/test (15% each)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"  ✓ Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
    
    # Step 3: Process and save images
    print("\n[3/4] Processing images (resizing to 128x128)...")
    
    splits = {
        'train': (train_images, train_labels, True),   # with augmentation
        'val': (val_images, val_labels, False),        # no augmentation
        'test': (test_images, test_labels, False)      # no augmentation
    }
    
    total_saved = 0
    
    for split_name, (images, labels, augment) in splits.items():
        print(f"\n  Processing {split_name} set...")
        
        for label_type in ['defect', 'no_defect']:
            output_dir = os.path.join(DATASET_ROOT, split_name, label_type)
            os.makedirs(output_dir, exist_ok=True)
            
            count = 0
            for image_path, label in zip(images, labels):
                if label == label_type:
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, filename)
                    
                    if preprocess_image(image_path, output_path, augment):
                        count += 1
                        total_saved += 1
            
            print(f"    ✓ {label_type}: {count} images")
    
    # Step 4: Summary
    print("\n[4/4] Summary")
    print("=" * 60)
    print(f"✓ Total images processed and saved: {total_saved}")
    print(f"\nDataset structure created at: {DATASET_ROOT}/")
    
    # Show dataset statistics
    for split in ['train', 'val', 'test']:
        defect_count = len(os.listdir(os.path.join(DATASET_ROOT, split, 'defect')))
        no_defect_count = len(os.listdir(os.path.join(DATASET_ROOT, split, 'no_defect')))
        total = defect_count + no_defect_count
        print(f"\n{split.upper()}:")
        print(f"  Defect:    {defect_count}")
        print(f"  No-defect: {no_defect_count}")
        print(f"  Total:     {total}")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete! Ready for training.")

if __name__ == '__main__':
    main()
