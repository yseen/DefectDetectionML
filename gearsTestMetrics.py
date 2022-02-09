# Quantify segmentation performance on test sets.

import os
import cv2
import numpy as np 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from pathlib import Path




true_list_new=[]
pred_list_new=[]

mask_path = Path("./Gear/Synthetic Images/11/mask").rglob("*.png")#use the last folder for test dataset
print("READING MASKS")
for im_mask_path in mask_path:
    print(im_mask_path)
    img = cv2.imread(str(im_mask_path))
    #img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    true_list_new.append(img)

pred_path = Path("./GearsTestSetOutput").rglob("*.png")#use the last folder for test dataset
print("READING PREDICTIONS")
for im_pred_path in pred_path:
    print(im_pred_path)
    img = cv2.imread(str(im_pred_path))
    #img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    pred_list_new.append(img)
    
print("Begin quantifying performance")
true_list_new=np.array(true_list_new)
true_list_new = np.where(true_list_new>127,1,0)
pred_list_new=np.array(pred_list_new)
pred_list_new = np.where(pred_list_new>127,1,0)

true_list_new=true_list_new.flatten()
pred_list_new=pred_list_new.flatten()

print("Confusion Matrix: ", 
      confusion_matrix(true_list_new, pred_list_new)) 

print ("Accuracy : ", 
       accuracy_score(true_list_new,pred_list_new)*100) 

print("Report : ", 
      classification_report(true_list_new, pred_list_new))

print("F1: ", f1_score(true_list_new, pred_list_new))

print("Jaccard:", jaccard_score(true_list_new, pred_list_new))

print("ROC_AUC:", roc_auc_score(true_list_new, pred_list_new))
