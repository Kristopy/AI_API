
# url = 'https://www.google.com/imgres?imgurl=https%3A%2F%2Fjohnloeber.com%2Fimages%2Fkmeans-mnist%2Fnines%2F4.png&imgrefurl=https%3A%2F%2Fjohnloeber.com%2Fdocs%2Fkmeans.html&tbnid=cUOjMzORgndilM&vet=10CAMQxiAoAGoXChMIiNartdT58wIVAAAAAB0AAAAAED4..i&docid=Xi4giF0PQiNgiM&w=237&h=237&itg=1&q=handwritten%20numbers%20images&ved=0CAMQxiAoAGoXChMIiNartdT58wIVAAAAAB0AAAAAED4'

import pathlib
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path().resolve(__file__)
DATASET = BASE_DIR / 'Datasets' / 'Datasets_NUM_REC'

IMAGES_DIR = DATASET / 'Images_convert'
IMAGES_PATH = IMAGES_DIR / 'Number_2.png'

image = cv2.imread(str(IMAGES_PATH), cv2.IMREAD_COLOR)
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_GRAY = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)


print('Original shape: ', image_GRAY.shape)
dim = (28,28)
resized = cv2.resize(image_GRAY, dim, interpolation=cv2.INTER_AREA)
print('Original shape: ', resized.shape)


print('Image vectorized: ', resized)

df = pd.DataFrame(resized)

df.to_csv("Test_2.csv", header=None,index=False)

cv2.imshow('Grey_image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def image_to_vector(image: numpy.ndarray) -> numpy.ndarray:
#     """
#     Args:
#     image: numpy array of shape (length, height, depth)

#     Returns:
#      v: a vector of shape (length x height x depth, 1)
#     """

#     length, height, depth = image.shape
#     return image.reshape((length * height * depth, 1))

