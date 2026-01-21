import tensorflow as tf
from tensorflow.keras import layers

class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(RecommenderModel, self).__init__()
        # Embeddings para usuarios
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        # Embeddings para items (películas/productos)
        self.item_embedding = layers.Embedding(num_items, embedding_size)
        # Capa de salida (Producto escalar)
        self.dot = layers.Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        return self.dot([user_vector, item_vector])

# Ejemplo de inicialización
model = RecommenderModel(num_users=1000, num_items=500, embedding_size=50)
model.compile(optimizer='adam', loss='mse')
print("Modelo de recomendación inicializado.")