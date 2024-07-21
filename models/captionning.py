from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings to clean up output

def show_n_generate(image_path : str):
    """
    Generate a text caption for an image using a pretrained Vision-Text Encoder-Decoder model.

    Parameters:
        image_path (str): The path to the image file to caption.

    Returns:
        str: A text caption generated based on the contents of the image.
    """

    # Load the image processing model and tokenizer from Hugging Face's model hub
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Open the image file and process it into the format expected by the model
    img = Image.open(image_path)
    pixel_values = image_processor(img, return_tensors="pt").pixel_values  # Convert image to model-compatible tensor

    # Load the vision-language encoder-decoder model
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Generate captions using the model; the output is a tensor of token ids
    generated_ids = model.generate(pixel_values, max_length=30)

    # Decode the token ids back to readable text while skipping special tokens like [PAD], [CLS]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


