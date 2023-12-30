from tensorflow.keras.preprocessing.text import Tokenizer

# Sample text data
sentences = [
    "ChatGPT is revolutionizing AI.",
    "Natural Language Processing is fascinating.",
    "Deep learning models are becoming more sophisticated."
]

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Fit the tokenizer on the text data
tokenizer.fit_on_texts(sentences)

# Get the word index
word_index = tokenizer.word_index

# Tokenize the sentences
sequences = tokenizer.texts_to_sequences(sentences)

print("Word Index:", word_index)
print("Sequences:", sequences)