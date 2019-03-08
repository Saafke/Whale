# -- Kaggle Whale --

import WhaleDataset, TripletLoss
from WhaleDataset import WhaleDataset
from TripletLoss import TripletLoss

import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler

resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
print('num_features:', num_ftrs)
print(resnet50.fc)
resnet50.fc = nn.Linear(num_ftrs, 128)

#modules = list(resnet50.children())[:-1]
#resnet50 = nn.Sequential(*modules)

for p in resnet50.parameters():
    p.requires_grad = True

resnet50.cuda()

# Init
lr = 0.002
lr_patience = 10
batch_size = 8
epochs = 100
tripletloss = TripletLoss(1)
save_name = "model1.pt"

# optimizer
optimizer = optim.Adam(resnet50.parameters(), lr=lr)

# scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=lr_patience, verbose=True, min_lr=0)

# dataset
whale_dataset = WhaleDataset(csv_file='/home/weber/Documents/Kaggle/Whale/triplets.cvs',
                                    image_dir='/home/weber/Documents/Kaggle/input/train', transform=True)

data_loader = DataLoader(dataset=whale_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         )

print("Training...")
# Let's train
for epoch in range(epochs):

    running_loss = 0
    index = 0

    # -- Loop over the training set in batches.
    for i, batch in enumerate(data_loader):
        # -- Get the batch
        anchor = batch["Anchor"].cuda()
        positive = batch["Positive"].cuda()
        negative = batch["Negative"].cuda()

        # -- Forward pass to get f(eature)vectors a.k.a. embedding vectors
        anchor_fvector = resnet50(anchor)
        pos_fvector = resnet50(positive)
        neg_fvector = resnet50(negative)

        #print("fvector size:", anchor_fvector.shape)
        # -- Calculate losses
        loss = tripletloss(anchor_fvector, pos_fvector, neg_fvector)

        # -- Update total loss for this 'nr' set
        running_loss += loss.item()

        # -- Backward pass and optimize
        optimizer.zero_grad()
        (loss).backward()
        optimizer.step()
        index = i

        if index % 100 == 0:
            print("index:", index)
    
    torch.save(resnet50, "./"+save_name)
    print("Epoch", epoch, "done. Average training loss:", running_loss / index)
    print("Model saved as", save_name)


