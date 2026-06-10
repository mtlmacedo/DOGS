import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), info = tfds.load(
    "stanford_dogs",
    split=["train[:10%]", "train[10%:12%]"],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = info.features["label"].num_classes

def preprocess(image, label):
    image = tf.image.resize(image, (224,224))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

ds_train = (
    ds_train
    .map(preprocess)
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

ds_test = (
    ds_test
    .map(preprocess)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        512,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=20
)

loss, acc = model.evaluate(ds_test)

print(acc)