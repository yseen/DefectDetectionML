# Quantify segmentation performance on real images.

#import tensorflow as tf
#from tensorflow import keras
from pathlib import Path
import torch
#import matplotlib.pyplot as plt
import cv2
import numpy as np
#import pandas as pd

print("LOADING MODEL")
# Load the trained model 
model = torch.load('./GearsModelE5B4/weights.pt')
# Set the model to evaluate mode
model.eval()
print("MODEL LOADED")



#measure gears
#path = Path("C:\\GitHub\DefectDetectionML\\Gear\\Synthetic Images\\11\\normal").rglob("*.png")#use the last folder for test dataset
path = Path("./Gear/Real Images/2resized/").rglob("*.png")#real images


print("GENERATING MASKS")
for img_path in path:
    print(img_path)
    # Read  an image from the data-set
    img = cv2.imread(str(img_path)).transpose(2,0,1).reshape(1,3,768,1024)
    
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
    print(a['out'].shape)
    #a['out'].data.cpu().numpy().flatten()
    #a['out'].cpu().detach().numpy()[0][0]>0.16
    output = np.where(a['out'].cpu().detach().numpy()[0][0]>0.1,255,0) #selected threshold through visual analysis
    #output = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)
    #convert output to RGB and save as PNG
    
    img2 = np.zeros((int(len(output[:,1])),int(len(output[1,:])),3))
    print(img2.shape)
    img2[:,:,0] = output
    img2[:,:,1] = output
    img2[:,:,2] = output
    output=img2
    #cv2.imwrite("C:\\GitHub\DefectDetectionML\\Gear\\Synthetic Images\\11\\predictions0\\"+img_path.name,output)
    cv2.imwrite("./Gear/Real Images/2predMask/"+img_path.name,output)