from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D , MaxPooling2D
from tensorflow.keras import backend as K
import os



# img = load_img('path/to/image.jpg')

train_data = r'C:\Users\MANASVI\Documents\GitHub\Emotion_detection\train'
test_data = r'C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator = train_datagen.flow_from_directory(
    train_data,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Validation generator
validation_generator = validation_datagen.flow_from_directory(
    test_data,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Class labels for emotions
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preview a batch
img, label = next(train_generator)

# CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))  # Softmax for multi-class classification

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Count number of images in train and test
num_train_imgs = sum([len(files) for _, _, files in os.walk(train_data)])
num_test_imgs = sum([len(files) for _, _, files in os.walk(test_data)])

print(f"Total Training Images: {num_train_imgs}")
print(f"Total Testing Images: {num_test_imgs}")

# Training the model
epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32
)

# Save the trained model
model.save('model_file.h5')