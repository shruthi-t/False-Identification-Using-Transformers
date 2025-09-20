import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# ----------------------
# Paths
# ----------------------
MODEL_PATH = "outputs/classifier"

# ----------------------
# Load model & tokenizer
# ----------------------
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Put model in evaluation mode
model.eval()

# ----------------------
# Prediction function
# ----------------------
def predict(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

    # Get predicted label
    predicted_id = torch.argmax(probs).item()
    confidence = probs[predicted_id].item()

    # Map id → label
    label = model.config.id2label[predicted_id]

    # Return nicely formatted result
    result = {
        "text": text,
        "predicted_label": label,
        "confidence": round(confidence, 3),
        "all_probs": {model.config.id2label[i]: round(p, 3) for i, p in enumerate(probs.tolist())}
    }
    return result


# ----------------------
# Example run
# ----------------------
if __name__ == "__main__":
    test_text = "Click this link to prevent your account from being locked"
    output = predict(test_text)

    print("\n🔍 Prediction Result")
    print(f"Text: {output['text']}")
    print(f"Predicted Label: {output['predicted_label']} (Confidence: {output['confidence']})")
    print("All Probabilities:", output["all_probs"])
