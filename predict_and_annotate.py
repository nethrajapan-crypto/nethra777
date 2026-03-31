"""
STEP 12 & 13: Predict on New Image & Annotate Output

Provides functionality to:
- Load a trained model
- Make predictions on single images
- Annotate images with prediction results
"""

import os
import cv2
import torch
from PIL import Image
from pathlib import Path

from model import create_model
from load_dataset import val_test_transform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './best_model.pth'
CLASS_NAMES = ['defect', 'no_defect']


def load_model(checkpoint_path=CHECKPOINT_PATH, model_name='efficientnet_b0'):
    """Load trained model from checkpoint."""
    model = create_model(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image_path, model=None):
    """Predict defect class on a single image.
    
    Args:
        image_path (str): path to input image
        model (torch.nn.Module): loaded model. Creates one if None.
    
    Returns:
        dict with 'class', 'confidence', 'probs'
    """
    if model is None:
        model = load_model()
    
    # load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = probs.max(1)
    
    pred_class = CLASS_NAMES[pred.item()]
    return {
        'class': pred_class,
        'confidence': confidence.item(),
        'probs': probs.squeeze().cpu().numpy()
    }


def annotate_image(image_path, output_path, model=None, draw_probs=True):
    """Annotate image with prediction and save.
    
    Args:
        image_path (str): input image path
        output_path (str): output annotated image path
        model (torch.nn.Module): model for prediction
        draw_probs (bool): include probability bar
    """
    if model is None:
        model = load_model()
    
    # predict
    result = predict_image(image_path, model)
    pred_class = result['class']
    confidence = result['confidence']
    
    # load image with cv2
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # draw prediction text
    text = f"{pred_class.upper()}: {confidence:.2%}"
    color = (0, 255, 0) if pred_class == 'no_defect' else (0, 0, 255)  # Green / Red
    
    cv2.putText(image, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # draw confidence bar
    if draw_probs:
        bar_width = w - 20
        bar_height = 30
        bar_y = h - 50
        
        # background bar
        cv2.rectangle(image, (10, bar_y), (10 + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        # confidence bar
        filled_width = int(bar_width * confidence)
        cv2.rectangle(image, (10, bar_y), (10 + filled_width, bar_y + bar_height), color, -1)
        
        # percentage text
        conf_text = f"{confidence:.1%}"
        cv2.putText(image, conf_text, (15, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # save
    cv2.imwrite(output_path, image)
    print(f"✓ Annotated image saved: {output_path}")


def compute_subtraction(template_path, test_path, output_path):
    """Generate subtraction map overlay and save."""
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)
    if template is None or test is None:
        print(f"Error reading template/test for subtraction: {template_path}, {test_path}")
        return False

    test_h, test_w = test.shape[:2]
    template = cv2.resize(template, (test_w, test_h))
    diff = cv2.absdiff(template, test)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(test, 0.7, heat, 0.3, 0)
    cv2.imwrite(output_path, overlay)
    print(f"✓ Subtraction image saved: {output_path}")
    return True


def batch_annotate(input_dir, output_dir, model=None):
    """Annotate all images in a directory.
    
    Args:
        input_dir (str): directory with input images
        output_dir (str): directory for annotated outputs
        model (torch.nn.Module): model for prediction
    """
    if model is None:
        model = load_model()
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f'annotated_{image_file}')
            
            try:
                annotate_image(input_path, output_path, model)
                count += 1
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    print(f"\nBatch annotation complete: {count} images processed")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_and_annotate.py <image_path> [output_path]")
        print("Example: python predict_and_annotate.py test_image.jpg output.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'annotated_output.jpg'
    
    model = load_model()
    result = predict_image(image_path, model)
    print(f"\nPrediction: {result['class']} (confidence: {result['confidence']:.2%})")
    print(f"Probabilities: defect={result['probs'][0]:.4f}, no_defect={result['probs'][1]:.4f}")
    
    annotate_image(image_path, output_path, model)
    print(f"Annotated image: {output_path}")
