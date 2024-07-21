import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

# Load the CLIP model (Vision Transformer B/32) and preprocessing tools, setting device based on GPU availability.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download and load the CIFAR-100 dataset, setting it up for evaluation (not training).
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def eval(image_path : str):
    """
    Evaluate an image by finding the most similar CIFAR-100 labels using CLIP model.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        list of tuples: List of (label, similarity percentage) for the top predictions where similarity > 50%.
    """
    # Open the image and preprocess it as per CLIP model requirements.
    img = Image.open(image_path)
    image_input = preprocess(img).unsqueeze(0).to(device)

    # Tokenize labels from CIFAR-100 and move them to the same device as the model.
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate image and text features using the CLIP model without updating the model parameters.
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize the feature vectors and compute the similarity as a softmax over dot products.
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Filter predictions with similarity greater than 50% and format the results.
    predictions = []
    for value, index in zip(values, indices):
        if value > 0.5:
            predictions.append((cifar100.classes[index], 100 * value.item()))

    return predictions
