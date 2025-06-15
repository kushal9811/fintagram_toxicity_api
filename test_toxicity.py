import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Load vectorizer (TFSMLayer model)
vectorizer = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.TFSMLayer("vectorizer", call_endpoint="serving_default")
])

# Load toxicity model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1800,), dtype=tf.float32),
    tf.keras.layers.TFSMLayer("toxicity", call_endpoint="serving_default")
])

def score_comment(text, threshold=0.5):
    # 1) Vectorize
    vect_dict = vectorizer(tf.constant([text]))
    vect = tf.cast(vect_dict["text_vectorization"], tf.float32)

    # 2) Call the toxicity model
    output_dict = model(vect, training=False)       # Returns a dict

    # 3) Grab the first tensor from the dict
    probs_tensor = list(output_dict.values())[0]    # e.g. next(iter(output_dict.values()))

    # 4) Convert to NumPy and take first example
    probs = probs_tensor.numpy()[0]

    # 5) Display
    for label, p in zip(LABELS, probs):
        print(f"{label:15s} → prob {p:.3f} → {'YES' if p > threshold else 'no'}")
    print()

# Sample inputs
examples = [
    "I love this! You are awesome.",
    "You freaking suck! I'm going to hit you.",
    "I will kill you, you piece of trash."
]

for ex in examples:
    print("Input:", ex)
    score_comment(ex)