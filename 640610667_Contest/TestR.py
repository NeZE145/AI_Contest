import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os


test_set_folder = "Test Set Samples"
test_set_CSV = test_set_folder +"\\test.csv"
test_set_images = test_set_folder +"\\Test Images/"

_model ="640610667_Contest\ResB0_2_Best_Loss.pth"
_csv = "testtest" +".csv" # name of the CSV that will be saved

class SiameseNetworkResB0(nn.Module):
    def __init__(self):
        super(SiameseNetworkResB0, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()
        self.fc = nn.Linear(1280, 1)

    def forward(self, img1, img2):
        feat1 = self.base_model(img1)
        feat2 = self.base_model(img2)
        score1 = self.fc(feat1)
        score2 = self.fc(feat2)
        return score1, score2

class FoodDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name1 = self.data.iloc[idx]["Image 1"]
        img_name2 = self.data.iloc[idx]["Image 2"]
        img_path1 = os.path.join(self.image_folder, img_name1)
        img_path2 = os.path.join(self.image_folder, img_name2)
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, img_name1, img_name2

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


df_test = pd.read_csv(test_set_CSV)
test_dataset = FoodDataset(df_test, test_set_images, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = SiameseNetworkResB0()
model.load_state_dict(torch.load(_model, map_location="cpu"))
model.eval()

results = []
with torch.no_grad():
    for img1, img2, img_name1, img_name2 in test_loader:
        output1, output2 = model(img1, img2)
        score1 = output1.item()
        score2 = output2.item()
        prediction = 1 if score1 > score2 else 2
        print(f"{img_name1[0]} vs {img_name2[0]} → Predicted: {prediction} Score1: {score1:.4f}, Score2: {score2:.4f}")

        results.append([img_name1[0], img_name2[0], prediction])

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Image 1", "Image 2", "Predicted Winner"])
results_df.to_csv(_csv, index=False)
print(f"Results saved to '{_csv}'.")