import os
import json
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping #Newly added
import numpy as np

# Load the existing model or create a new one if it doesn't exist
model_path = 'Final_Model_50_Epochs.keras'
if not os.path.exists(model_path):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    model = load_model(model_path)

# Image data generation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'                                  
)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Load and iterate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Bobby Black\Documents\AI_Dataset\Training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True)  # Set shuffle to True

validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\Bobby Black\Documents\AI_Dataset\Validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True)  # Set shuffle to True

# Calculate steps_per_epoch based on the number of batches you want to process per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)



# Train model with early stopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Number of images in the training set
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps)  # Number of images in the validation set
    #callbacks=[early_stopping]) #Newly added

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()



# Predictions
predictions = model.predict(validation_generator)
predicted_classes = np.round(predictions).astype(int).flatten()  # Convert to binary class labels
true_classes = validation_generator.classes

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)


report = classification_report(true_classes, predicted_classes, target_names=validation_generator.class_indices)
print(report)


val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {val_accuracy*100:.2f}%, Validation loss: {val_loss}")


# Save the updated model
model.save(model_path)

# Save training history to a JSON file
with open('training_history.json', 'w') as json_file:
    json.dump(history.history, json_file)


# Load training history from a JSON file
with open('training_history.json', 'r') as json_file:
    loaded_history = json.load(json_file)

# Now, loaded_history is a dictionary containing the training history

