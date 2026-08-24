import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IClevrDataset(Dataset):
    def __init__(self, root_dir, json_file, objects_json):
        self.root_dir = root_dir

        with open(objects_json, 'r') as f:
            self.classes = json.load(f)
        with open(json_file, 'r') as f:
            self.data_dict = json.load(f)
            
        self.img_names = list(self.data_dict.keys())

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        labels = self.data_dict[img_name]
        
        one_hot = torch.zeros(24)
        for label in labels:
            class_idx = self.classes[label]
            one_hot[class_idx] = 1.0
            
        return image, one_hot
    

if __name__ == '__main__':
    try:
        test_dataset = IClevrDataset(
            root_dir='./iclevr', 
            json_file='./data/train.json', 
            objects_json='./data/objects.json'
        )
        
        print(f"資料集總共有 {len(test_dataset)} 張圖片")
        
        img, label = test_dataset[0]
        
        print("\n--- 檢查第一筆資料 ---")
        print(f"影像維度 (Shape): {img.shape}")
        print(f"影像數值範圍: min={img.min():.4f}, max={img.max():.4f}")
        print(f"標籤維度 (Shape): {label.shape}")
        print(f"標籤 One-hot 內容:\n{label}")
        
    except FileNotFoundError as e:
        print(f"找不到檔案，請確認路徑是否正確: {e}")