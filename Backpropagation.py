import streamlit as st
import tensorflow as tf
import numpy as np

def main():
    st.title("Neural Network Classifier")

    # Generate synthetic data for demonstration
    np.random.seed(0)
    X_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, size=100)

    # Define the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, activation='sigmoid', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model using backpropagation
    model.fit(X_train, y_train, epochs=100)

    # Test the model with new data
    X_test = np.array([[0.1, 0.2], [0.8, 0.9]])
    predictions = model.predict(X_test)

    st.write("Predictions for test data:")
    st.write(predictions)

if __name__ == "__main__":
    main()
