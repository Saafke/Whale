import torch, os
import pandas as pd
import numpy as np
import kdtree
from PIL import Image
from torchvision import transforms, utils
### Generate feature vectors and make submission file
## Sources -- https://github.com/stefankoegl/kdtree -- ##

# This class emulates a tuple, but contains a useful payload
class Item(object):
    def __init__(self, vector , img_filename, id):
        self.vector = vector
        self.img_filename = img_filename
        self.id = id

    def __getid__(self):
        return id

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, i):
        return self.vector[i]

    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.img_filename, self.id, self.vector)

def gen_fvectors():
    # -- Loop through all images and generate f_vectors
    df = pd.read_csv("/home/weber/Documents/Kaggle/input/train.csv")
    df["f_vector"] = ""
    df = df[df["Id"] != 'new_whale']
    print(df.head())

    fvector_dict = {}

    for index, row in df.iterrows():
    # open image as resized tensor
        cur_image_name = row['Image']
        cur_id = row['Id']
        cur_image = Image.open(os.path.join(image_dir, cur_image_name)).convert('RGB')
        cur_image_tensor = transform(cur_image).cuda().unsqueeze(0)

        # get f_vector/embedding from model
        with torch.no_grad(): # don't save gradients bro
            f_vector = model(cur_image_tensor)

        # put fvector/embedding (w corresponding id) into dict
        fvector_dict[cur_image_name] = f_vector.cpu().numpy(),cur_id

    print(fvector_dict)
    # save dict
    np.save("./fvector_dict", fvector_dict) # can't save this shit to a pd df

def make_submission(kdtree_bool):
    # make submission dataframe
    df = pd.read_csv("/home/weber/Documents/Kaggle/input/sample_submission.csv")
    df = df.drop(columns=['Id'])
    df['Id'] = ''

    fvector_dict = np.load("/home/weber/Documents/Kaggle/Whale/fvector_dict.npy")

    print(df.head())
    
    if kdtree_bool:
        #### create kdtree 
        fvector_kdtree = kdtree.create(dimensions=2048)
        
        # put all precalculated fvectors in the kdtree 
        for key, value in fvector_dict.item().items(): #key: image filename | value : tensor, id
            # make vector
            current_ground_fvector = np.squeeze(value[0])
            # make item for tree
            item = Item(current_ground_fvector, key, value[1])
            # add this item to kdtree
            fvector_kdtree.add(item)

        print("kd-tree made.")

        # loop over submission images
        for index, row in df.iterrows():
            cur_image = row['Image']
            cur_id = row['Id']
            cur_image = Image.open(os.path.join(test_image_dir, cur_image)).convert('RGB')
            cur_image_tensor = transform(cur_image).cuda().unsqueeze(0)

            # get f_vector/embedding from model
            with torch.no_grad(): # don't save gradients bro
                f_vector = model(cur_image_tensor)
            
            # make item for kdtree
            fvector_np = np.squeeze(f_vector.cpu().numpy())
            item = Item(fvector_np, cur_image, cur_id)
            
            # get nn in kdtree
            nn = fvector_kdtree.search_nn( item )
            nn_id = nn[0].data.id
            print("nearest neighbour id:", nn_id , "\n")
            
            # set closest id
            df.at[index, 'Id'] = nn_id + " new_whale" + " new_whale" + " new_whale" + " new_whale"

        df.to_csv('./submission_by_kdtree.csv', index=False)
        ####

    else:

    #### linear 
        # loop through submission to get test_filenames of whales
        for index, row in df.iterrows():
            cur_image = row['Image']
            cur_image = Image.open(os.path.join(test_image_dir, cur_image)).convert('RGB')
            cur_image_tensor = transform(cur_image).cuda().unsqueeze(0)

            # get f_vector/embedding from model
            with torch.no_grad(): # don't save gradients bro
                f_vector = model(cur_image_tensor)

            smallest_dist = float('inf')
            closest_whale_id = ''

            # compare predicted feature vector with all pre-calculated train vectors 
            for key, value in fvector_dict.item().items(): #key: image filename | value : tensor, id
                current_ground_fvector = torch.from_numpy(value[0]).cuda()
                
                # get the distance from test_fvector to current precalculated_fvector
                dist = torch.dist(f_vector, current_ground_fvector)

                # get 5 id of whales that are closest
                if dist < smallest_dist:
                    smallest_dist = dist
                    closest_whale_id = value[1]

            # set closest id
            df.at[index, 'Id'] = closest_whale_id + " new_whale" + " new_whale" + " new_whale" + " new_whale"

            if index % 1 == 0:
                print(index, "/", df.shape[0])

        df.to_csv('./submission.csv', index=False)
    ####

submission = pd.read_csv("/home/weber/Documents/Kaggle/Whale/submission.csv")
submission = submission.drop(submission.columns[0], axis=1)
submission.to_csv('./submission_noindex.csv', index=False)
print("done")

while(True):
    pass
# init
transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])
image_dir = '/home/weber/Documents/Kaggle/input/train'
test_image_dir = '/home/weber/Documents/Kaggle/input/test'

# Load trained model
model = torch.load("/home/weber/Documents/Kaggle/Whale/model1.pt")
model.eval()

# Generate all feature vectors
#gen_fvectors()

# Get nearest neighbours - make submission file
use_kdtree = False
make_submission(use_kdtree)





