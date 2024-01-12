import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from fcn import FCN
from resunet import ResUNet
from data_load import ToothDataset
import os
from torchvision import transforms
from PIL import Image



# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += images.size(0)

            # 计算准确率
            with torch.no_grad():
                predictions = (outputs > 0.4).float()
                correct = (predictions == masks).float()
                accuracy = correct.sum() / correct.numel()
                total_correct += accuracy.item()

            progress_bar.set_postfix({'Loss': total_loss / total_samples, 'Accuracy': total_correct / total_samples})

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        # print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

def test_model(model, loader, save_dir):
    model = model.to(device)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, mask) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            # 假设选择通道 index 为 1 作为牙齿的通道
            teeth_channel = 1
            segmentation_mask = (outputs.argmax(0) == teeth_channel).float()
            # Convert model output to segmentation mask
            segmentation_mask = (outputs.argmax(1) == teeth_channel).float()

            # Apply threshold if needed
            segmentation_mask = (segmentation_mask > 0.5).float()

            # Save the predictions with the same name and format as the original image
            img_name = unlabelled_train_dataset.image_paths[i]
            save_path = os.path.join(save_dir, img_name)
            print(save_path)

            # Convert predictions to PIL image without resizing
            predictions_pil = transforms.ToPILImage()(segmentation_mask.squeeze().cpu())

            # Save the PIL image
            predictions_pil.save(save_path)


if __name__ == '__main__':
    # 数据目录
    data_dir = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_train\train'
    data_dir2 = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_image'

    # 数据转换
    transform = transforms.Compose([transforms.ToTensor()])

    # 训练集
    labelled_train_image_folder = os.path.join(data_dir, 'labelled', 'image')
    labelled_train_mask_folder = os.path.join(data_dir, 'labelled', 'label')
    labelled_train_dataset = ToothDataset(labelled_train_image_folder, mask_folder=labelled_train_mask_folder,
                                          transform=transform)
    unlabelled_image_folder = os.path.join(data_dir, 'unlabelled', 'image')
    unlabelled_train_dataset = ToothDataset(unlabelled_image_folder, transform=transform)

    # 测试集
    test_image_folder = os.path.join(data_dir2, 'image')
    test_dataset = ToothDataset(test_image_folder, transform=transform)

    # 加载模型
    # model = UNet(1,1)
    # model = FCN(1,1)
    model = ResUNet(1,1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoader for training
    train_loader = DataLoader(labelled_train_dataset, batch_size=1, shuffle=True)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # DataLoader for testing
    test_loader = DataLoader(unlabelled_train_dataset, batch_size=1)

    # Test the model and save predictions
    test_save_dir = r"D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_train\train\unlabelled\label"
    test_model(model, test_loader, test_save_dir)


