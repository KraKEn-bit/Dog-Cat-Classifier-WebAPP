import tensorflow as tf
import os

old_name = 'dog_cat_final_model.keras'
new_name = 'dog_cat_fixed.keras'

if os.path.exists(old_name):
    print(f"Loading {old_name}...")
    model = tf.keras.models.load_model(old_name, compile=False)
    model.save(new_name)
    print(f"Successfully created: {new_name}")
else:
    print(f"Error: Could not find {old_name} in this folder.")