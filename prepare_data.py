import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from matplotlib.pyplot import imshow
from IPython.display import HTML


##### Data #####

# -- Set paths
img_train_path = os.path.abspath('../input/train')
img_test_path = os.path.abspath('../input/test')
csv_train_path = os.path.abspath('../input/train.csv')

# -- Load dataframe
df = pd.read_csv(csv_train_path)
print(df.head())
print("Size (with new_whales):", df['Image'].size, "\n")

# -- Remove 'new_whale' instances/rows
df = df[df["Id"] != 'new_whale']
print(df.head())
print("Size (without new_whales):", df['Image'].size)

# -- Count unique whale Id's
unique_ids = df['Id'].nunique()
print("Number of unique Ids/whales:", unique_ids, "\n")

# -- Count number of occurences per whale Id
print("Number of occurences per whale (.head()):")
print(df.Id.value_counts().head())

# -- Let's make triplets
# -- new data frame
triplets = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative', 'Switch'])

print("Preparing triplets...")
# -- Loop through all images, and make triplets
for index, row in df.iterrows():
    cur_image = row['Image']
    cur_id = row['Id']
    switch = False
    
    # Get a random positive 
    pos_df = df[(df.Id == cur_id) & (df.Image != cur_image)]
    if pos_df.empty == True: # only one image of this whale
        positive = cur_image
        switch = True
    else:
        positive = pos_df.sample(1).Image.values[0]
    

    # Get a random negative
    negative = df[(df.Id != cur_id)].sample(1).Image.values[0]

    # Make a triplet and add it to all triplets
    triplets = triplets.append({'Anchor' : cur_image , 'Positive' : positive, 'Negative': negative, 'Switch': switch} , ignore_index=True)

print(triplets.head())
triplets.to_csv("./triplets.csv")
print("Triplets successfully save to triplets.csv.")




