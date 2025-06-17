import tensorflow as tf

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

vectorizer = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.TFSMLayer("vectorizer", call_endpoint="serving_default")
])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1800,), dtype=tf.float32),
    tf.keras.layers.TFSMLayer("toxicity", call_endpoint="serving_default")
])

def score_comment(text, threshold=0.5):
    vect = tf.cast(vectorizer(tf.constant([text]))["text_vectorization"], tf.float32)
    probs = list(model(vect, training=False).values())[0].numpy()[0]

    for label, p in zip(LABELS, probs):
        print(f"{label:15s} → prob {p:.3f} → {'YES' if p > threshold else 'no'}")
    print()

examples = [
    "I love this! You are awesome.",
    "You freaking suck! I'm going to hit you.",
    "I will kill you, you piece of trash."
]

for ex in examples:
    print("Input:", ex)
    score_comment(ex)