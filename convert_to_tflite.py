import tensorflow as tf

# --- Convert toxicity model ---
tox_conv = tf.lite.TFLiteConverter.from_saved_model("toxicity")
tox_conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tox_conv._experimental_lower_tensor_list_ops = False
tox_conv.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_tox = tox_conv.convert()
open("toxicity/toxicity.tflite", "wb").write(tflite_tox)

# --- Convert vectorizer model ---
vec_conv = tf.lite.TFLiteConverter.from_saved_model("vectorizer")
vec_conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
vec_conv.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_vec = vec_conv.convert()
open("vectorizer/vectorizer.tflite", "wb").write(tflite_vec)

print("✅ TFLite models written:")
print("   - toxicity/toxicity.tflite")
print("   - vectorizer/vectorizer.tflite")
