import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model Architecture:
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


try:
    model.load_weights('dog_cat_final_model.keras')
    # 3. Save it as a fresh Keras 3 native file
    model.save('dog_cat_fixed.keras')
    print("✅ Success! Model weights transferred to dog_cat_fixed.keras")
except Exception as e:
    print(f"❌ Error transferring weights: {e}")