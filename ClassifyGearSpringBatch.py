# Quantify classification performance on test sets.

import tensorflow as tf
from tensorflow import keras
from pathlib import Path

print("LOADING MODEL")
model = keras.models.load_model('C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\classE1.pt')

print("MODEL LOADED")

img = keras.preprocessing.image.load_img(
    "C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\test\\springs\\render_2.png", target_size = (224, 224)
)
#measure gears
gear_count=0
classified_count=0
path = Path("C:\\GitHub\DefectDetectionML\\ClassificationDataSet\\test\\gears").rglob("*.png")
print("GEAR CLASSIFICATION")
for img_path in path:
    print(img_path)
    img = keras.preprocessing.image.load_img(img_path, target_size = (224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    if score<=0.5:
        classified_count +=1
        print("Gear. Correct")
    else:
        print("Spring. INCORRECT.")
    gear_count +=1

print("Total gears: ", gear_count)
print("correctly classified gears: ", classified_count)

#measure springs
spring_count=0
classified_count=0
path = Path("C:\\GitHub\DefectDetectionML\\ClassificationDataSet\\test\\springs").rglob("*.png")
print("SPRING CLASSIFICATION")
for img_path in path:
    print(img_path)
    img = keras.preprocessing.image.load_img(img_path, target_size = (224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    if score>0.5:
        classified_count +=1
        print("Spring. Correct")
    else:
        print("Gear. INCORRECT.")
    spring_count +=1

print("Total springs: ", spring_count)
print("correctly classified gears: ", classified_count)
