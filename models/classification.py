import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def eval(image):
    img=Image.open(image)
    image_input = preprocess(img).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

   # Store predictions with similarity > 50% in a vector
    predictions = []
    for value, index in zip(values, indices):
        if value > 0.5:  # Check if similarity is greater than 50%
            predictions.append((cifar100.classes[index], 100 * value.item()))

    return predictions

print(eval('./chat.jpg'))
