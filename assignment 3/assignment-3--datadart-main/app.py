# app.py
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define SEQ_LENGTH
SEQ_LENGTH = 112

# Load the model and necessary mappings
model = load_model('model_1_shakespeare.h5')
with open("shakespeare.txt", "r", encoding="utf-8") as file:
     text = file.read()

# Unique characters in the dataset
chars = sorted(list(set(text)))
num_chars = len(chars)

# Create character to index and index to character mappings
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# Function to generate text
def generate_text(seed_text, next_chars=200):
    num_chars = len(char_to_index)  # Define num_chars here
    generated_text = seed_text
    for _ in range(next_chars):
        X_pred = np.zeros((1, SEQ_LENGTH * num_chars))
        for t, char in enumerate(seed_text):
            X_pred[0, t * num_chars + char_to_index[char]] = 1
        preds = model.predict(X_pred, verbose=0)[0]
        next_index = np.random.choice(num_chars, p=preds)
        next_char = index_to_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

def visualize_embeddings():
    embeddings = model.layers[0].get_weights()[0]
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
    plt.title('PCA Visualization of Character Embeddings')
    plt.xlabel('Character Diversity')
    plt.ylabel('Character Variation')

    # Annotate points with characters
    for i, char in enumerate(chars):
        plt.annotate(char, (embeddings_pca[i, 0], embeddings_pca[i, 1]))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Streamlit app
def main():
    st.title('Text Prediction App')
    input_text = st.text_input('Enter some text:', 'The quick brown fox jumps over the lazy dog')
    k = st.number_input('Number of characters to predict:', min_value=1, max_value=200, value=10, step=1)

    if st.button('Predict'):
        prediction = generate_text(input_text, next_chars=k)
        st.write('Predicted text:', prediction)
    
    if st.checkbox('Visualize Embeddings'):
        visualize_embeddings()

if __name__ == '__main__':
    main()