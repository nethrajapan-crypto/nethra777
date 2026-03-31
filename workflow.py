"""
COMPLETE WORKFLOW GUIDE
Module 4: Evaluation & Prediction Testing

This script demonstrates the complete end-to-end workflow for training,
evaluating, and making predictions with the PCB defect detection model.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False
    print(f"\n✓ {description} completed")
    return True


def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  PCB DEFECT DETECTION - COMPLETE WORKFLOW                ║")
    print("║  Module 1-4: Data Preparation → Evaluation               ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    steps = [
        ("python train.py", 
         "STEP 7-8: Train Model (10 epochs, save best checkpoint)"),
        
        ("python evaluate.py",
         "STEP 14: Generate Full Evaluation Report"),
        
        ("python report_generator.py",
         "STEP 14+: Create HTML Visualization Report"),
        
        ("python predict_and_annotate.py dataset/test/defect/*.jpg output_annotated.jpg",
         "STEP 12-13: Predict on New Image & Annotate (example)"),
    ]
    
    completed = 0
    failed = 0
    
    for cmd, desc in steps:
        if run_command(cmd, desc):
            completed += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Completed: {completed}")
    print(f"✗ Failed: {failed}")
    
    if failed == 0:
        print(f"\n{'='*60}")
        print("📊 OUTPUTS GENERATED")
        print(f"{'='*60}")
        print("✓ best_model.pth - Trained model weights")
        print("✓ training_metrics.json - Training history")
        print("✓ evaluation_report/ - Full evaluation metrics")
        print("  ├── evaluation_report.json - Detailed metrics (JSON)")
        print("  ├── confusion_matrix.png - Confusion matrix visualization")
        print("  ├── training_curves.png - Loss & accuracy plots")
        print("  └── roc_curve.png - ROC curve")
        print("✓ report.html - Interactive HTML report")
        print("✓ output_annotated.jpg - Predicted image with annotation")
        print(f"\n{'='*60}")
        print("✨ Pipeline Complete! Open report.html to view results.")
        print(f"{'='*60}")
    else:
        print("\n⚠️  Some steps failed. Check the errors above.")
        sys.exit(1)


def quick_reference():
    """Print quick reference guide."""
    print("""
    
╔═══════════════════════════════════════════════════════════╗
║  QUICK REFERENCE - MODULE 4 COMPONENTS                  ║
╚═══════════════════════════════════════════════════════════╝

📌 STEP 11: Load Trained Model for Inference
   Usage: 
   from predict_and_annotate import load_model
   model = load_model('./best_model.pth')

📌 STEP 12: Predict on Single Image
   Usage:
   from predict_and_annotate import predict_image
   result = predict_image('test_image.jpg', model)
   print(result['class'], result['confidence'])

📌 STEP 13: Annotate Output Image
   Usage:
   from predict_and_annotate import annotate_image
   annotate_image('test.jpg', 'output.jpg', model)

📌 STEP 14: Generate Evaluation Report
   Usage:
   python evaluate.py
   
   Outputs:
   - evaluation_report/evaluation_report.json
   - evaluation_report/confusion_matrix.png
   - evaluation_report/training_curves.png
   - evaluation_report/roc_curve.png

📌 Generate HTML Report
   Usage:
   python report_generator.py
   
   Output: report.html

───────────────────────────────────────────────────────────

KEY METRICS IN REPORT:
  • Accuracy: Overall correct predictions
  • Precision: True Defects / Detected as Defects
  • Recall: Detected Defects / Actual Defects
  • F1-Score: Harmonic mean of Precision & Recall
  • ROC-AUC: Classification threshold performance
  • False Positives: No-Defect flagged as Defect
  • False Negatives: Defect flagged as No-Defect

───────────────────────────────────────────────────────────

FILES GENERATED:
  ✓ best_model.pth - Best model checkpoint
  ✓ training_metrics.json - Training history for plotting
  ✓ evaluation_report.json - Full evaluation metrics
  ✓ Confusion matrix, ROC curve, training curves (PNG)
  ✓ report.html - Interactive visualization report
  ✓ annotated_*.jpg - Predicted output images
    """)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        quick_reference()
    else:
        main()
