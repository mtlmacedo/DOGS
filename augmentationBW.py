import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), info = tfds.load(
    "stanford_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = info.features["label"].num_classes

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

def preprocess(image, label):

    image = tf.image.resize(image, (224,224))

    if tf.random.uniform([]) < 0.3:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)

    image = augmentation(image)
    image = image / 255.0

    return image, label

ds_train = ds_train.map(preprocess).batch(32)
ds_test = ds_test.map(
    lambda x,y: (tf.image.resize(x,(224,224))/255.0,y)
).batch(32)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(ds_train, epochs=20)

loss, acc = model.evaluate(ds_test)

print("Accuracy:", acc)