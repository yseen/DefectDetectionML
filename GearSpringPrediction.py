#program to read in an image, determine if it is a gear or spring, call the relevant defect detection segmentation model and save the result to file.
#sample call 1: python .\GearSpringPrediction.py --input_dir '.\Gear\Synthetic Images\11\normal\render_1.png' --output_dir .\outMask.png
#sample call 2: python .\GearSpringPrediction.py --input_dir '.\Spring\Synthetic Images\5\normal\render_10.png' --output_dir .\outMask2.png
import click
import tensorflow as tf
from tensorflow import keras
import torch
import cv2
import numpy as np

@click.command()
@click.option("--input_dir",
              required=True,
              help="Specify the input image path.")
@click.option("--output_dir",
              required=True,
              help="Specify the output image path.")

def main(input_dir, output_dir):
    print("LOADING CLASSIFICATION MODEL")
    model = keras.models.load_model('C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\classE1.pt')
    print("MODEL LOADED")
    img = keras.preprocessing.image.load_img(input_dir, target_size = (224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    print(predictions)
    score = predictions[0]
    if score>0.5: #spring
        print("Image is spring.\nLOADING SPRINGS DEFECT DETECTION MODEL")
        # Load the trained model 
        model = torch.load('./SpringsModelE4B4/weights.pt')
        # Set the model to evaluate mode
        model.eval()
        print("MODEL LOADED")
        img = cv2.imread(str(input_dir)).transpose(2,0,1).reshape(1,3,768,1024)
        with torch.no_grad():
            a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
        output = np.where(a['out'].cpu().detach().numpy()[0][0]>0.1,255,0) #selected threshold through visual analysis
    else: #gear
        print("Image is gear.\nLOADING GEARS DEFECT DETECTION MODEL")
        # Load the trained model 
        model = torch.load('./GearsModelE5B4/weights.pt')
        # Set the model to evaluate mode
        model.eval()
        print("MODEL LOADED")
        img = cv2.imread(str(input_dir)).transpose(2,0,1).reshape(1,3,768,1024)
        with torch.no_grad():
            a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
        output = np.where(a['out'].cpu().detach().numpy()[0][0]>0.12,255,0) #selected threshold through visual analysis
    img2 = np.zeros((int(len(output[:,1])),int(len(output[1,:])),3))
    img2[:,:,0] = output
    img2[:,:,1] = output
    img2[:,:,2] = output
    output=img2
    cv2.imwrite(output_dir,output)
    print("Image mask saved. Goodbye!")

if __name__ == "__main__":
    main()


'''
print("LOADING CLASSIFICATION MODEL")
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
'''
