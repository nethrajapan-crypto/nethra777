
"""
STEP 14: Comprehensive Evaluation Report

Generate a detailed evaluation report including:
- Confusion matrix
- Classification metrics (precision, recall, F1)
- ROC-AUC analysis
- False positive/negative analysis
- Training curves visualization
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)
import torch

from load_dataset import make_dataloader
from model import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './best_model.pth'
METRICS_PATH = './training_metrics.json'
CLASS_NAMES = ['defect', 'no_defect']


def get_all_predictions(model, loader):
    """Get predictions and labels for entire dataset."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved: {output_path}")
    return cm


def plot_training_curves(metrics_path='training_metrics.json', output_path='training_curves.png'):
    """Plot training and validation loss/accuracy curves."""
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
    ax1.plot(epochs, metrics['val_loss'], 'r-o', label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, metrics['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
    ax2.plot(epochs, metrics['val_acc'], 'r-o', label='Val Acc', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Training curves saved: {output_path}")


def plot_roc_curve(y_true, y_probs, output_path='roc_curve.png'):
    """Plot ROC curve for binary classification."""
    # Get probabilities for positive class (defect = 0)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 0])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ ROC curve saved: {output_path}")
    return roc_auc


def analyze_errors(y_true, y_pred, y_probs):
    """Analyze false positives and false negatives."""
    fp_mask = (y_true != y_pred) & (y_pred == 1)  # False positives
    fn_mask = (y_true != y_pred) & (y_pred == 0)  # False negatives
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    
    analysis = {
        'false_positives': {
            'count': int(fp_count),
            'avg_confidence': float(y_probs[fp_mask, 1].mean()) if fp_count > 0 else 0,
            'min_confidence': float(y_probs[fp_mask, 1].min()) if fp_count > 0 else 0,
            'max_confidence': float(y_probs[fp_mask, 1].max()) if fp_count > 0 else 0,
        },
        'false_negatives': {
            'count': int(fn_count),
            'avg_confidence': float(y_probs[fn_mask, 0].mean()) if fn_count > 0 else 0,
            'min_confidence': float(y_probs[fn_mask, 0].min()) if fn_count > 0 else 0,
            'max_confidence': float(y_probs[fn_mask, 0].max()) if fn_count > 0 else 0,
        }
    }
    return analysis


def compute_defect_type_accuracy(y_true, y_pred, file_paths, defect_types=None):
    """Compute accuracy for each defect type based on file name patterns."""
    if defect_types is None:
        defect_types = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    out = {}

    # Ensure file_paths aligns with y_true/y_pred
    if len(file_paths) != len(y_true):
        raise ValueError('Length of file_paths must match y_true')

    for defect in defect_types:
        mask = (y_true == 0) & np.array([defect in os.path.basename(p) for p in file_paths])
        count = int(mask.sum())
        out[defect] = {
            'accuracy': float((y_pred[mask] == 0).mean()) if count > 0 else None,
            'count': count,
        }

    return out


def print_predictions(file_paths, y_true, y_pred, y_probs, max_rows=20):
    """Print predictions and confidence scores to the terminal."""
    total = len(y_true)
    # If max_rows is None or < 0, show all rows
    if max_rows is None or max_rows < 0:
        max_rows = total

    max_rows = min(total, max_rows)

    print("\nPredictions (first {} of {}):".format(max_rows, total))
    print("file\ttrue\tpred\tconfidence")

    for i in range(max_rows):
        fname = os.path.basename(file_paths[i])
        true_label = CLASS_NAMES[y_true[i]]
        pred_label = CLASS_NAMES[y_pred[i]]
        conf = float(y_probs[i, y_pred[i]])
        print(f"{fname}\t{true_label}\t{pred_label}\t{conf:.3f}")

    if max_rows < total:
        print(f"... (showing first {max_rows} of {total} samples)")


def generate_report(output_dir='evaluation_report', show_predictions=False, max_predictions=20, defect_type=None):
    """Generate comprehensive evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model = create_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    
    loaders, sizes = make_dataloader()
    
    # Get test predictions
    print("Generating predictions on test set...")
    y_true, y_pred, y_probs = get_all_predictions(model, loaders['test'])

    file_paths = [p for p, _ in loaders['test'].dataset.samples]

    # If requested, compute metrics for a specific defect type
    defect_accuracy = None
    if defect_type:
        defect_mask = (np.array([defect_type in os.path.basename(p) for p in file_paths]) & (y_true == 0))
        defect_count = int(defect_mask.sum())
        if defect_count > 0:
            defect_accuracy = float((y_pred[defect_mask] == 0).mean())
        else:
            defect_accuracy = None
    
    # Create report
    report = {
        'dataset': {
            'total_samples': int(sizes['test']),
            'defect_samples': int((y_true == 0).sum()),
            'no_defect_samples': int((y_true == 1).sum()),
        },
        'model': {
            'architecture': 'EfficientNet-B0',
            'pretrained': True,
        },
        'test_metrics': {
            'accuracy': float((y_pred == y_true).mean()),
            'roc_auc': float(roc_auc_score(y_true, y_probs[:, 0])),
        },
        'error_analysis': analyze_errors(y_true, y_pred, y_probs),
    }
    
    # Add per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True,
                                        target_names=CLASS_NAMES)
    report['class_metrics'] = class_report

    if defect_type:
        report['defect_type'] = defect_type
        report['defect_type_accuracy'] = {
            'accuracy': defect_accuracy,
            'count': defect_count,
        }

    # Add per-defect-type accuracy
    defect_types = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    per_defect_accuracy = compute_defect_type_accuracy(y_true, y_pred, file_paths, defect_types)
    report['per_defect_accuracy'] = per_defect_accuracy

    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, 
                               os.path.join(output_dir, 'confusion_matrix.png'))
    report['confusion_matrix'] = cm.tolist()
    
    # Plot training curves
    plot_training_curves(METRICS_PATH, 
                        os.path.join(output_dir, 'training_curves.png'))
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(y_true, y_probs,
                            os.path.join(output_dir, 'roc_curve.png'))

    # Optionally print per-sample predictions and confidences
    if show_predictions:
        print_predictions(file_paths, y_true, y_pred, y_probs, max_rows=max_predictions)
    
    # Save report to JSON
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 60)
    print(f"Test Set Size: {sizes['test']} samples")
    print(f"  Defects: {(y_true == 0).sum()} | No-defects: {(y_true == 1).sum()}")
    print(f"\nAccuracy: {report['test_metrics']['accuracy']:.4f}")
    print(f"ROC-AUC: {report['test_metrics']['roc_auc']:.4f}")
    print(f"\nFalse Positives: {report['error_analysis']['false_positives']['count']}")
    print(f"False Negatives: {report['error_analysis']['false_negatives']['count']}")

    if defect_type:
        dt = report.get('defect_type_accuracy', {})
        if dt.get('count', 0) > 0:
            print(f"\nAccuracy for defect type '{defect_type}': {dt['accuracy']:.4f} ({dt['count']} samples)")
        else:
            print(f"\nAccuracy for defect type '{defect_type}': N/A (no samples)")

    # Per-defect-type accuracy
    print("\nPer-defect-type accuracy:")
    for defect, stats in report['per_defect_accuracy'].items():
        if stats['count'] == 0:
            print(f"  {defect:16s}: N/A (no samples)")
        else:
            print(f"  {defect:16s}: {stats['accuracy']:.4f} ({stats['count']} samples)")

    print(f"\nClassification Report:\n{json.dumps(class_report, indent=2)}")
    print("\n" + "=" * 60)
    print(f"Report saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    defect_types = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

    parser = argparse.ArgumentParser(description='Evaluate model and print metrics.')
    parser.add_argument('--output-dir', default='evaluation_report', help='Directory to save report outputs.')
    parser.add_argument('--show-predictions', action='store_true', help='Print per-sample predictions and confidence scores.')
    parser.add_argument('--max-predictions', type=int, default=20, help='Maximum number of per-sample rows to print when using --show-predictions. Use -1 to show all.')
    parser.add_argument('--defect-type', choices=defect_types, help='If set, print and report metrics for this specific defect type.')
    args = parser.parse_args()

    max_preds = args.max_predictions if args.max_predictions >= 0 else None
    generate_report(output_dir=args.output_dir,
                    show_predictions=args.show_predictions,
                    max_predictions=max_preds,
                    defect_type=args.defect_type)
