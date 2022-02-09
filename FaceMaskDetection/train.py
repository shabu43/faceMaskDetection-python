# required packages
# pip install --upgrade tensorflow
# pip install --upgrade keras
# pip install --upgrade sklearn
# pip install  --upgrade imutils
# pip install  --upgrade opencv-python
# pip install  --upgrade scipy
# pip install  --upgrade numpy
# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import os
import numpy as np

# initialization of the learning rate
INIT_LR = 1e-4
EPOCHS = 20
B_size = 32

DIRECTORY = r"C:\Users\habib\Desktop\mask5\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# getting image from directory to data list
data = []
labels = []
i = 0

print("Images being loaded from directory...")
while i < len(CATEGORIES):
    j = 0
    dir = os.path.join(DIRECTORY, CATEGORIES[i])
    while j < len(os.listdir(dir)):
        img_path = os.path.join(dir, os.listdir(dir)[j])
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)

        data.append(img)
        labels.append(CATEGORIES[i])
        j = j + 1
    i = i + 1
# convert levels(masked, without mask) to Binary
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="uint8")
labels = np.array(labels)

# separating the training and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
NUM_TRAIN_IMAGES = len(trainX)
NUM_TEST_IMAGES = len(testX)
# training image generator for data augmentation
# https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# lconvolutional neural network, MobileNetV2
# https://keras.io/api/applications/mobilenet/
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Create head and the base model
# https://github.com/JetBrains/KotlinDL/issues/300
top_model = baseModel.output
top_model = AveragePooling2D(pool_size=(7, 7))(top_model)
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(128, activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

# Call top and the base model
train_model = Model(inputs=baseModel.input, outputs=top_model)

# loop over all layers in the base model and freeze them
i = 0
while i < len(baseModel.layers):
    baseModel.layers[i].trainable = False
    i = i + 1

# compiling the model
print("Model Compilation is ongoing...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
train_model.compile(loss="binary_crossentropy", optimizer=opt,
                    metrics=["accuracy"])

#https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# train the top of the neural network
print("Training is ongoing...")
H = train_model.fit(
    aug.flow(trainX, trainY, batch_size=B_size),
    steps_per_epoch=NUM_TRAIN_IMAGES // B_size,
    validation_data=(testX, testY),
    validation_steps=NUM_TEST_IMAGES // B_size,
    epochs=EPOCHS)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
predIdxs = train_model.predict(testX, batch_size=B_size)
predIdxs = np.argmax(predIdxs, axis=1)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# classification report
print("Classification report saved in report.text")
with open('report.txt', 'w') as f:
    f.write(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

#save the model
train_model.save("mask_detector.model", save_format="h5")
print("model saved to the directory")
