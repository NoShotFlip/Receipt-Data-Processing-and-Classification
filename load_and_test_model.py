import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image

# Load the pre-trained model and processor
model_name_or_path = "nielsr/layoutlmv3-finetuned-cord"
processor = AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=True)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

# Switch the model to evaluation mode
model.eval()


def prepare_inputs(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Use the processor's OCR functionality to handle text and bounding boxes automatically
    inputs = processor(images=image, return_tensors="pt")

    return inputs


def predict(image_path):
    # Prepare inputs
    inputs = prepare_inputs(image_path)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Map the predicted labels to the actual label names
    predicted_labels = [
        model.config.id2label[label_id.item()] for label_id in predictions[0]
    ]

    # Print the predictions
    print("Predicted labels:", predicted_labels)


if __name__ == "__main__":
    image_path = "./test.jpeg"  # Replace this with the path to your image
    predict(image_path)
