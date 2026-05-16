# debug_model.py
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('vgg16_best.keras', compile=False)
print(f"Couches : {len(model.layers)}")
print(f"Output shape : {model.output_shape}")

# Test avec image noire
test = np.zeros((1, 224, 224, 3), dtype=np.float32)
pred = model.predict(test, verbose=0)[0]
print(f"Prédiction image noire : {pred}")

# Test avec image blanche
test2 = np.ones((1, 224, 224, 3), dtype=np.float32)
pred2 = model.predict(test2, verbose=0)[0]
print(f"Prédiction image blanche : {pred2}")

# Test avec image random
test3 = np.random.rand(1, 224, 224, 3).astype(np.float32)
pred3 = model.predict(test3, verbose=0)[0]
print(f"Prédiction image random : {pred3}")