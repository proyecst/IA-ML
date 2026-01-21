import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# 1. Cargar el Tokenizer y el Modelo pre-entrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 2. Preparar una frase de ejemplo
text = "I absolutely love how this neural network performs!"
inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=64)

# 3. Predicción
outputs = model(inputs)
logits = outputs.logits
prediction = tf.nn.softmax(logits, axis=-1)

print(f"Probabilidades (Negativo, Positivo): {prediction.numpy()}")