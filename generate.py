import os
import re
import json
import numpy as np
import tensorflow as tf



MODEL_PATH = "models/lstm_text_generator.keras"
VOCAB_PATH = "models/vocabulary.json"


SEQ_LENGTH = 30

OUTPUT_DIR = "output"
OUTPUT_FILE = "output/generated_text.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading vocabulary...")

model = tf.keras.models.load_model(MODEL_PATH)

print("model loaded successfully.")


with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

word_to_id = vocabulary["word_to_id"]

id_to_word = {int(v): k for k, v in word_to_id.items()}

def generate_text(seed_text, num_words=50, temperature=0.8):
    seed_text = seed_text.lower()
    seed_words = re.sub(r'[^a-zA-Z0-9\s]', '', seed_text)
    seed_words = re.sub(r'\s+', ' ', seed_words).strip()
    seed_words = seed_words.split()

    unknown_words = [word for word in seed_words if word not in word_to_id]
    if unknown_words:
        print("Unknown words in seed text:", unknown_words)

        seed_words = [word for word in seed_words if word in word_to_id]

    if not seed_words:
        return "no valid seed words found in vocabulary."

    generated_words = seed_words.copy()

    for _ in range(num_words):
        current_words = generated_words[-SEQ_LENGTH:]
        sequence = [word_to_id[word] for word in current_words]
        if len(sequence) < SEQ_LENGTH:
            sequence = [0] * (SEQ_LENGTH - len(sequence)) + sequence

        input_sequence = np.array(sequence).reshape(1, SEQ_LENGTH)
        predictions = model.predict(input_sequence, verbose=0)[0]
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
        next_id = np.random.choice(len(predictions), p=predictions)
        next_word = id_to_word.get(int(next_id), None)
        if next_word is None:
            generated_words.append("<UNK>")
        else:
            generated_words.append(next_word)


    return " ".join(generated_words)

def save_generated_text(text, output_dir=OUTPUT_DIR, output_file=OUTPUT_FILE):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Generated text saved to {output_path}")

result = generate_text(seed_text="Once upon a time", num_words=50, temperature=0.8)
save_generated_text(result)


if __name__ == "__main__":
    print("\n")


    print("lstm text generator is ready to generate text based on your seed input."
    )

    while True:
        seed_text = input("\nEnter seed text (or type 'exit' to quit): ")
        if seed_text.lower() == 'exit':
            break

        num_words = input("Enter number of words to generate (default 50): ")
        try:
            num_words = int(num_words)
        except ValueError:
            num_words = 50

        temperature = input("Enter temperature (default 0.8): ")
        try:
            temperature = float(temperature)
        except ValueError:
            temperature = 0.8

        result = generate_text(seed_text, num_words=num_words, temperature=temperature)
        print("\nGenerated text:\n", result)



