#to solve "OSError: image file is truncated" error in classification training set
# find the image(s) that are causing the error. Images that are truncated, i.e. don't have the full pixels from header
#Outcome: Springs orig_3_20.png is broken (truncated)

import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image
from io import StringIO 

path = Path("C:\\GitHub\DefectDetectionML\\ClassificationDataSet\\train\\springs").rglob("*.png")
for img in path:
    pilImg = PIL.Image.open(img)
    try:
        pilImg.load()
    except OSError:
        print(img)