import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
from test import (
    GalaxyConvNeXt, GalaxyEfficientNet4,
    predict_galaxies, StarGalaxyEffNet
)

# Device setup
device = torch.device("cpu")  # change to 'cuda' if GPU available

# Load models (once)
star_galaxy_checkpoint = "Models/StarGalaxy_CLassifier/checkpoint.pth"
convnext_checkpoint = "Models/UGRIZ/model.pth"
efficientnet4_checkpoint = "Models/EfficientNet_Morphology2/model.pth"

# Star/Galaxy model
star_galaxy_model = StarGalaxyEffNet(num_outputs=2).to(device)
checkpoint = torch.load(star_galaxy_checkpoint, map_location=device)
state_dict = checkpoint['model_state_dict']
filtered_state = {k: v for k, v in state_dict.items() if k in star_galaxy_model.state_dict()}
star_galaxy_model.load_state_dict({**star_galaxy_model.state_dict(), **filtered_state})
star_galaxy_model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Main processing function
def process(filepath):
    """
    Process an astronomy image through the pipeline:
    Star/Galaxy -> ugriz -> morphology (if galaxy)
    
    Args:
        filepath (str): path to uploaded image
    
    Returns:
        dict: results including object_type, ugriz, morphology
    """
    results = {}

    # Load image
    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Step 1: Star/Galaxy classification
    with torch.inference_mode():
        output = star_galaxy_model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
        # convert to percentages rounded to 2 decimals
        star_galaxy_percent = [round(p * 100, 2) for p in probs]
        pred_class = "galaxy" if probs.argmax() == 0 else "star"

        results['object_type'] = pred_class
        results['star_galaxy_probs'] = {
            "galaxy": star_galaxy_percent[0],
            "star": star_galaxy_percent[1]
        }
        results['star_galaxy_most_likely'] = pred_class

    # Step 2: ugriz prediction (keep as is)
    ugriz_result = predict_galaxies(convnext_checkpoint, [filepath], device=device,
                                   model_class=GalaxyConvNeXt)[0]
    results['ugriz'] = [ugriz_result[f'out_{i}'] for i in range(5)]

    # Step 3: Morphology prediction if galaxy
    if pred_class == "galaxy":
        morph_result = predict_galaxies(efficientnet4_checkpoint, [filepath],
                                        device=device, model_class=GalaxyEfficientNet4)[0]
        morph_probs = [round(m * 100, 2) for m in [morph_result[f'out_{i}'] for i in range(4)]]
        class_labels = ["elliptical", "spiral", "edgeon", "merger"]
        results['morphology_probs'] = dict(zip(class_labels, morph_probs))
        results['morphology_most_likely'] = class_labels[morph_probs.index(max(morph_probs))]
    else:
        results['morphology_probs'] = None
        results['morphology_most_likely'] = None

    return results


# Optional: test from command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = process(sys.argv[1])
        print(res)
