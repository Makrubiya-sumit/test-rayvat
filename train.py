import os
import re
import numpy as np
import tensorflow as tf


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


DATA_PATH = "data/shakespeare.txt"
MODEL_DIR = "models"
MODEL_PATH = "models/lstm_text_generator.keras"

VOCAB_PATH = "models/vocabulary.json"

SEQ_LENGTH = 30

EMBEDDING_DIM = 128
LSTM_UNITS_1 = 256
LSTM_UNITS_2 = 256
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
MAX_TRAIN_WORDS = 150000
MAX_VALIDATION_WORDS = 30000




os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading and preprocessing data...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
    print("original text length:", len(text))


print("processing text...")
text = text.lower()

start_marker=("*** START OF THIS PROJECT GUTENBERG EBOOK  ***")

end_marker=("*** END OF THIS PROJECT GUTENBERG EBOOK  ***")

start_position = text.find(start_marker)
end_position = text.find(end_marker)


if start_position != -1 and end_position != -1:
    text = text[start_position + len(start_marker):end_position]
    print("text length after removing header and footer:", len(text))



text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
text = re.sub(r'\s+', ' ', text).strip()

print("cleaned text length:", len(text))

print("\nTokenizing text...")

words = text.split()

print("total words:", len(words))

print("unique words:", len(set(words)))

word_to_id = { word:index+1 for index, word in enumerate(set(words)) }
id_to_word = { index: word for word, index in word_to_id.items() }


vocab_size = len(word_to_id) + 1

print("vocabulary size:", vocab_size)

vocabulary = {
    "word_to_id": word_to_id,
    "id_to_word": {str(k): v for k, v in id_to_word.items()}
}


with open(VOCAB_PATH, "w", encoding="utf-8") as file:
    import json
    json.dump(vocabulary, file, ensure_ascii=False, indent=4)

print("Vocabulary saved to", VOCAB_PATH)

token_ids =np.array([word_to_id[word] for word in words], dtype=np.int32)

split_index = int(len(token_ids) * 0.90)

train_tokens = token_ids[:split_index]
validation_tokens = token_ids[split_index:]


print("train tokens length:", len(train_tokens))
print("validation tokens length:", len(validation_tokens))
train_tokens = train_tokens[:MAX_TRAIN_WORDS]
validation_tokens = validation_tokens[:MAX_VALIDATION_WORDS]




def create_sequences(tokens, seq_length):
    inputs = []
    targets = []

    for i in range(len(tokens) - seq_length):
       inputs_sequence = tokens[i:i + seq_length]
       target_word = tokens[i + seq_length]
       inputs.append(inputs_sequence)
       targets.append(target_word)
    return np.array(inputs,dtype=np.int32), np.array(targets,dtype=np.int32)


print("Creating training sequences...")

X_train, y_train = create_sequences(train_tokens, SEQ_LENGTH)

print("Creating validation sequences...")


X_val, y_val = create_sequences(validation_tokens, SEQ_LENGTH)

print("Training sequences shape:", X_train.shape, y_train.shape)
print("Training targets shape:", y_train.shape)
print("Validation sequences shape:", X_val.shape, y_val.shape)



print("\n Creating tensorflow datasets...")


train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE))

validation_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE))



print("\nBuilding the model...")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding"),
    LSTM(LSTM_UNITS_1, return_sequences=True, name="lstm_1"),
    Dropout(0.2, name="dropout_1"),
    LSTM(LSTM_UNITS_2, name="lstm_2"),
    Dropout(0.2, name="dropout_2"),
    Dense(vocab_size, activation='softmax', name="output")
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nmodel architecture:")

model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

print("\n Starting model training...")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[early_stopping, checkpoint])


print("\n")
print("=*60")


print("Training completed. Model saved to", MODEL_PATH)

print("\n")
print("=*60")

print("Vocabulary saved to", VOCAB_PATH)



