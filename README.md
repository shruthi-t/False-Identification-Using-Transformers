AI Fraud Detection using DistilBERT

This project is an end-to-end NLP system that detects fraudulent messages using a fine-tuned DistilBERT transformer model. It classifies text into four fraud behavior categories: Manipulation, Impersonation, Trust Building, and Distraction/Obfuscation.

The system demonstrates a complete machine learning pipeline:

Raw Text → Tokenization → DistilBERT Fine-Tuning → Model Inference → Prediction

Project Structure:

False Detection using Transformer

* data

  * data.csv
  * evidence_docs.csv
* src

  * train_classifier.py
  * infer.py
  * build_store.py
  * rag_sniffer.py
  * app.py
* outputs

  * classifier

    * model.safetensors
    * config.json
    * tokenizer_config.json
    * vocab.txt
* requirements.txt
* README.md

Setup:

Create a virtual environment (optional):

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Train the Model:

Fine-tune DistilBERT on the fraud dataset:

python src/train_classifier.py

This will load data from data/data.csv, tokenize text using the DistilBERT tokenizer, fine-tune the distilbert-base-uncased model, and save the trained model into the outputs/classifier directory.

Run Inference:

Use the trained model to classify new text:

python src/infer.py

Example output:

Prediction Result
Text: Click this link to prevent your account from being locked
Predicted Label: manipulation (Confidence: 0.512)
All Probabilities: {'manipulation': 0.512, 'impersonation': 0.187, 'trust_building': 0.13, 'distraction': 0.171}

Highlights:

This project fine-tunes a DistilBERT transformer for fraud detection, implements a custom PyTorch dataset and tokenizer pipeline, supports multi-class classification with confidence scores, and provides CLI-based inference for real-time predictions.
It is extendable with retrieval-based evidence using a RAG-style approach. The project showcases practical experience with transformer models, NLP pipelines, and real-world fraud detection use cases.
