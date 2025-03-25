import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os


Train_images_folder = "Questionair Images/Questionair Images"  #Folder of images that will be used for training.
Train_CSV = "data_from_questionaire.csv" 

model_name ="ResnetB0"


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Use a pretrained EfficientNet as the feature extractor
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()  

        self.fc = nn.Linear(1280, 1)  # EfficientNet-B0 outputs 1280 features

    def forward(self, img1, img2):
        feat1 = self.base_model(img1)
        feat2 = self.base_model(img2)

        score1 = self.fc(feat1)
        score2 = self.fc(feat2)

        return score1, score2

class FoodDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name1 = self.data.iloc[idx]["Image 1"]
        img_name2 = self.data.iloc[idx]["Image 2"]
        winner = self.data.iloc[idx]["Winner"]

        # Construct full image paths
        img_path1 = os.path.join(self.image_folder, img_name1)
        img_path2 = os.path.join(self.image_folder, img_name2)

        label = 1 if winner == 1 else 0

        # Load images
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_folder = Train_images_folder 
dataset = FoodDataset(csv_file=Train_CSV, image_folder=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


class PairwiseRankingLoss(nn.Module):
    def __init__(self):
        super(PairwiseRankingLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, score1, score2, labels):
        logits = score1 - score2
        loss = self.loss_fn(logits, labels.view(-1, 1).float())  # Ensure correct shape
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)
criterion = PairwiseRankingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20



total_loss_best = float('inf')  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for img1, img2, labels in dataloader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        score1, score2 = model(img1, img2)
        loss = criterion(score1, score2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)

    if avg_loss < total_loss_best:
        total_loss_best = avg_loss
        torch.save(model.state_dict(), model_name +"Best_Loss.pth")
        print("best loss model saved")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), model_name+".ptn")