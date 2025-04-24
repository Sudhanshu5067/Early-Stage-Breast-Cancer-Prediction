from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-20:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'breast_cancer/train',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    'breast_cancer/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train = train_generator.n // train_generator.batch_size
history = model.fit(train_generator, epochs=10, verbose=1, validation_data=test_generator)

model.save("alg/FinalModel.h5")

from matplotlib import pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'], label='Testing accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('# Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("alg/model2_acc.png")

plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'], label='Testing loss', color='green')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('# Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("alg/model2_loss.png")

acc = history.history['accuracy'][-1]
print(acc)
