from PIL import Image
from resizeimage import resizeimage
import os
import glob

# first crop to square in center of image
#then resize square to desired size



if not os.path.exists('C:/Users/toon1/Documents/Private Reading Spring 2018/AllImages'):
    os.makedirs('C:/Users/toon1/Documents/Private Reading Spring 2018/AllImages')

dir = 'C:/Users/toon1/Documents/Private Reading Spring 2018/Chestnut'
count = 1
for file in os.listdir(dir):
    filename = str(file)
    img = Image.open(dir + "/" + filename)
    image_size = (224, 224)
    width, height = img.size

    if width > height:
       delta = width - height
       left = int(delta/2)
       upper = 0
       right = height + left
       lower = height
    else:
       delta = height - width
       left = 0
       upper = int(delta/2)
       right = width
       lower = width + upper

    img = img.crop((left, upper, right, lower))
    img = img.resize(image_size, Image.ANTIALIAS)
    filename = "chestnut_" + str(count)
    fileName = "C:/Users/toon1/Documents/Private Reading Spring 2018/AllImages/" + filename
    img.save(fileName, 'jpeg')
    print("done " + str(count))
    count += 1
    img.close()
