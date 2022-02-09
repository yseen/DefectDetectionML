#to solve "PIL.UnidentifiedImageError: cannot identify image file" error in classification training set
# find the image(s) that are causing the error
#Outcome: Springs orig_4_25.png is broken

import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image

path = Path("C:\\GitHub\DefectDetectionML\\ClassificationDataSet\\train\\gears").rglob("*.png")
for img_p in path:
    try:
        img = Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)