import tensorflow as tf
from tensorflow import keras

print("LOADING MODEL")
model = keras.models.load_model('C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\classE1.pt')

print("MODEL LOADED")

img = keras.preprocessing.image.load_img(
    "C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\test\\springs\\render_2.png", target_size = (224, 224)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
print(predictions)
score = predictions[0]
print(
    "This image is %.2f percent gear and %.2f percent spring."
    % (100 * (1 - score), 100 * score)
)

print ("Predicted: ", tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1))

