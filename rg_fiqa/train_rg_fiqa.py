import os
import cv2
import torch
import argparse
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

# Add project root to path to allow direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rg_fiqa import RGFIQA
from recognize.process_nb116_person_proxyfusion_quality import get_comprehensive_quality_score, app

class FQDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img_bgr = cv2.imread(p)
        score, face = 0, None
        if img_bgr is not None:
            faces = app.get(img_bgr)
            if faces:
                face = max(faces, key=lambda f: f.det_score)
            score = get_comprehensive_quality_score(img_bgr, face)
        else:
            # Handle case where image is not read correctly
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), torch.tensor([score], dtype=torch.float32)

    def __len__(self):
        return len(self.img_paths)

def collect_images(root):
    out = []
    for r, _, files in os.walk(root):
        out += [os.path.join(r, f) for f in files if f.endswith('.jpg')]
    return out

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    all_imgs = collect_images(args.data_root)
    random.shuffle(all_imgs)
    
    val_split = int(len(all_imgs) * 0.1)
    train_imgs = all_imgs[val_split:]
    val_imgs = all_imgs[:val_split]

    print(f"Dataset split: {len(train_imgs)} for training, {len(val_imgs)} for validation.")

    train_ds = FQDataset(train_imgs, transform)
    val_ds = FQDataset(val_imgs, transform)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    net = RGFIQA().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(args.epochs):
        net.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {ep+1}/{args.epochs} [Training]"):
            x, y = x.to(device), y.to(device)
            pred = net(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        
        avg_train_loss = train_loss / len(train_ds)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {ep+1}/{args.epochs} [Validation]"):
                x, y = x.to(device), y.to(device)
                pred = net(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)
        
        avg_val_loss = val_loss / len(val_ds)

        print(f"Epoch {ep+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), "checkpoints/rg_fiqa_best.pth")
            print(f"  -> New best model saved with validation loss: {best_val_loss:.6f}")

    torch.save(net.state_dict(), "checkpoints/rg_fiqa_last.pth")
    print("\nTraining complete. Last epoch model saved to 'rg_fiqa_last.pth'.")
    print(f"Best validation model saved to 'rg_fiqa_best.pth' with loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train RG-FIQA: Rule-Guided Face Image Quality Assessment.")
    pa.add_argument("--data_root", default=r"C:\Project\Classroom-Reid\dataset\NB116_person_spatial", help="Root directory of the dataset.")
    pa.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    pa.add_argument("--bs", type=int, default=128, help="Batch size for training.")
    pa.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    pa.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. Set to 0 on Windows if you face issues.")
    main(pa.parse_args())
