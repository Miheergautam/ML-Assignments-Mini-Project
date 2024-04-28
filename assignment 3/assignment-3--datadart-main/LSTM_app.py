# Streamlit app
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
# Function to preprocess text and create sequences for training
def preprocess_text(text, seq_length):
    chars = sorted(list(set(text)))
    char_to_index = {char: i for i, char in enumerate(chars)}
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, 1):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        sequences.append([char_to_index[char] for char in seq])
        next_chars.append(char_to_index[target])
    return np.array(sequences), np.array(next_chars), chars, char_to_index

# Load the Shakespeare dataset
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Preprocess the text and create sequences for training
SEQ_LENGTH = 100
X, y, chars, char_to_index = preprocess_text(text, SEQ_LENGTH)
num_chars = len(chars)
index_to_char = {i: char for char, i in char_to_index.items()}

# Function to generate text using the trained model
def generate_text(model, seed_text, char_to_index, index_to_char, seq_length=100, num_chars=200):
    generated_text = seed_text
    for _ in range(num_chars):
        # Pad the seed text if its length is less than the sequence length
        while len(seed_text) < seq_length:
            seed_text = " " + seed_text
        X_pred = np.zeros((1, seq_length))
        for t, char in enumerate(seed_text):
            X_pred[0, t] = char_to_index[char]
        pred = model.predict(X_pred, verbose=0)[0]
        next_index = np.random.choice(len(pred), p=pred)
        next_char = index_to_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

def visualize_embeddings(embeddings, chars):
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

model = load_model('model_LSTM.h5')

def main():
    st.title('Text Prediction App')
    input_text = st.text_input('Enter some text:', 'The quick brown fox jumps over the lazy dog')
    k = st.number_input('Number of characters to predict:', min_value=1, max_value=200, value=10, step=1)

    if st.button('Predict'):
        prediction = generate_text(model, input_text, char_to_index, index_to_char, num_chars=k)
        st.write('Predicted text:', prediction)

    # if st.checkbox('Visualize Embeddings'):
    #     embeddings = model.layers[0].get_weights()[0]
    #     visualize_embeddings(embeddings, chars)

if __name__ == '__main__':
    main()