import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class PatchedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("batch_shape", None)
        super(PatchedInputLayer, self).__init__(**kwargs)

print(f"TF Version: {tf.__version__}")
try:
    print("Loading with PatchedInputLayer...")
    model = tf.keras.models.load_model(
        "agrimater_model2.h5", 
        custom_objects={"InputLayer": PatchedInputLayer},
        compile=False
    )
    print("Success loading model!")
    print(model.summary())
except Exception as e:
    print(f"Error: {e}")
