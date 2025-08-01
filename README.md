# Sentiment_Analysis_by_RNN

# Sentiment Analysis using Simple RNN

This notebook demonstrates a basic sentiment analysis task using a Simple Recurrent Neural Network (RNN) with the IMDB movie reviews dataset.

## Overview

The goal of this notebook is to classify movie reviews as either positive or negative based on their text content.

## Steps Performed

1.  **Library Imports:** Necessary libraries like TensorFlow, Keras, NumPy, Matplotlib, and Pandas are imported.
2.  **Parameter Definition:** Vocabulary size and maximum sequence length are defined.
3.  **Dataset Loading:** The IMDB movie reviews dataset is loaded using `tf.keras.datasets.imdb.load_data()`. The dataset is pre-tokenized and comes with a vocabulary index.
4.  **Data Preprocessing:** The sequences are padded to a fixed length (`mx_l`) using `preprocessing.sequence.pad_sequences()` to ensure uniform input size for the RNN.
5.  **Model Creation:** A Simple RNN model is created using the Keras Sequential API.
    *   An Embedding layer is used to convert word indices into dense vectors.
    *   A SimpleRNN layer processes the embedded sequences.
    *   A Dense layer with a sigmoid activation function outputs the probability of the review being positive.
6.  **Model Compilation:** The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.
7.  **Model Training:** The model is trained on the training data for a specified number of epochs and batch size. Validation data is used to monitor performance during training.
8.  **Model Evaluation:** The trained model is evaluated on the test dataset to assess its performance on unseen data.
9.  **Word Index Mapping:** The word-to-index mapping is retrieved and a reverse mapping (index-to-word) is created.
10. **Text Encoding Function:** A function `en_text` is defined to preprocess and encode new text reviews into the format required by the model.
11. **Sentiment Prediction:** The `en_text` function is used to encode example reviews (including negative and sarcastic ones), and the trained model is used to predict their sentiment. The predictions (probabilities) are then converted into sentiment labels (positive or negative).

## Usage

To use this notebook:

1.  Run all the code cells sequentially.
2.  Modify the `reviews_list` in the prediction cell to analyze the sentiment of your own reviews.

## Model Performance

The current Simple RNN model achieves an accuracy of approximately 50% on the test dataset. This indicates that the model is not performing significantly better than random chance, likely due to the simplicity of the model architecture for this complex task and the nuances of sentiment in text (e.g., sarcasm).

