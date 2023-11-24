{\rtf1\ansi\ansicpg1252\cocoartf2757
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 !jupyter nbconvert --to script preprocessDefinition.ipynb #nb to py\
import tensorflow as tf\
from tensorflow import keras\
import tensorflow_datasets as tfds\
from tensorflow.keras import layers\
from preprocessDefinition import prep\
\
#model def\
def create_model(input_shape, num_classes):\
    base_model = keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)\
    \
    #fine tune more layer, can accuracy increase some more?\
    for layer in base_model.layers[:-30]:\
        layer.trainable = False\
\
    model = keras.Sequential([\
        base_model,\
        layers.GlobalAveragePooling2D(),\
        layers.Dense(num_classes, activation='softmax')\
    ])\
\
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),  # Adjust the learning rate\
                  loss='sparse_categorical_crossentropy',\
                  metrics=['accuracy'])\
\
    return model\
\
model_name_flowers = 'xception'\
train_s, info = tfds.load(name='oxford_flowers102', split='train+validation', as_supervised=True, with_info=True)\
valid_s = tfds.load(name='oxford_flowers102', split='test[90%:]', as_supervised=True)\
test_s = tfds.load(name='oxford_flowers102', split='test[:90%]', as_supervised=True)\
\
train_p = train_s.map(lambda image, label: prep(image, label, model_name_flowers), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32).prefetch(tf.data.experimental.AUTOTUNE)\
valid_p = valid_s.map(lambda image, label: prep(image, label, model_name_flowers), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32).prefetch(tf.data.experimental.AUTOTUNE)\
test_p = test_s.map(lambda image, label: prep(image, label, model_name_flowers), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32).prefetch(tf.data.experimental.AUTOTUNE)\
\
# call model def\
input_shape = info.features['image'].shape\
num_classes = info.features['label'].num_classes\
model = create_model(input_shape, num_classes)\
\
#data aug to fix over fitting\
data_augmentation = keras.Sequential([\
    layers.experimental.preprocessing.RandomFlip("horizontal"),\
    layers.experimental.preprocessing.RandomRotation(0.1),\
    layers.experimental.preprocessing.RandomZoom(0.1),\
])\
\
train_p_augmented = train_p.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
\
#train w data augmentation\
epochs = 25\
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)\
out = model.fit(train_p_augmented, epochs=epochs, validation_data=valid_p, callbacks=[early_stopping], verbose=2)\
\
#eval\
model.evaluate(test_p)\
model.save('flowersModel.keras')}