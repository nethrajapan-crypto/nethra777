"""
STEP 5: Load EfficientNet Model

Utility for creating an EfficientNet classifier using the `timm` library.
Supports loading pretrained weights and adjusting the final head to the
binary defect / no-defect task.
"""

import timm
import torch.nn as nn


def create_model(model_name='efficientnet_b0', num_classes=2, pretrained=True):
    """Instantiate an EfficientNet model from timm.

    Args:
        model_name (str): timm model identifier (e.g. 'efficientnet_b0').
        num_classes (int): number of output classes. default 2 (defect/no-defect)
        pretrained (bool): load ImageNet pretrained weights.

    Returns:
        torch.nn.Module
    """
    # Try to create model with pretrained weights first
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
    except Exception as e:
        # If pretrained fails, try without pretrained
        print(f"⚠ Warning: Could not load pretrained weights: {e}")
        print(f"  Creating model without pretrained weights...")
        model = timm.create_model(model_name, pretrained=False)

    # detect classifier head name, replace with new linear layer
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # fallback: try to find last linear layer automatically
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                parent = model
                # traverse path to module to replace it
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], nn.Linear(in_features, num_classes))
                break
    return model


if __name__ == '__main__':
    # quick sanity check
    m = create_model()
    print(m)
    sample = torch.randn(1, 3, 128, 128)
    out = m(sample)
    print('Output shape:', out.shape)
