import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from fcn import FCN  # 假设你的 FCN 模型定义在 fcn.py 中
from resunet import ResUNet
from u_net import UNet2D
# from ENet import UNetENet
from mynet import myUNet2D
from data_load import ToothDataset
import matplotlib.pyplot as plt

# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")


def visualize_last_image(model, loader):
    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i == len(loader) - 1:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                # Apply threshold if needed
                segmentation_mask = (outputs > 0).float()
                # Convert predictions to PIL image without resizing
                predictions_pil = transforms.ToPILImage()(segmentation_mask.squeeze().cpu())

                # Display the original image, ground truth mask, and predicted mask
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(images.squeeze().cpu().numpy(), cmap='gray')
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(masks.squeeze().cpu().numpy(), cmap='gray')
                plt.title('Ground Truth Mask')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(predictions_pil, cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.show()

                break

# Train the model
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
#     model = model.to(device)
#     model.train()
#
#     best_accuracy = 0.0
#
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
#
#         for i, (images, masks) in enumerate(progress_bar):
#             images, masks = images.to(device), masks.to(device)
#
#             optimizer.zero_grad()
#
#             outputs = model(images)
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_samples += images.size(0)
#
#             # 计算准确率
#             with torch.no_grad():
#                 predictions = (outputs > 0).float()
#                 correct = (predictions == masks).float()
#                 accuracy = correct.sum() / correct.numel()
#                 total_correct += accuracy.item()
#
#             progress_bar.set_postfix({'Loss': total_loss / total_samples, 'Accuracy': total_correct / total_samples})
#
#         avg_loss = total_loss / total_samples
#         avg_accuracy = total_correct / total_samples
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')
#
#         # 在每个 epoch 结束后进行验证
#         val_accuracy = validate_model(model, val_loader, criterion)
#
#         # 保存性能最好的模型
#         if val_accuracy > best_accuracy:
#             best_accuracy = val_accuracy
#             torch.save(model.state_dict(), 'best_model.pth')
#
#         # 学习率调度器步进
#         scheduler.step()
#         # 在每轮结束后可视化最后一张图片及其分割结果
#         visualize_last_image(model, train_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    model = model.to(device)
    model.train()

    best_accuracy = 0.0

    train_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

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
                predictions = (outputs > 0).float()
                correct = (predictions == masks).float()
                accuracy = correct.sum() / correct.numel()
                total_correct += accuracy.item()

            progress_bar.set_postfix({'Loss': total_loss / total_samples, 'Accuracy': total_correct / total_samples})

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

        # 在每个 epoch 结束后进行验证
        val_accuracy = validate_model(model, val_loader, criterion)

        train_loss_list.append(avg_loss)
        train_accuracy_list.append(avg_accuracy)
        val_accuracy_list.append(val_accuracy)

        # 保存性能最好的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        # 学习率调度器步进
        scheduler.step()
        # 在每轮结束后可视化最后一张图片及其分割结果
        visualize_last_image(model, train_loader)

    # Plot the training curves
    plot_training_curves(train_loss_list, train_accuracy_list, val_accuracy_list, num_epochs)

def validate_model(model, val_loader, criterion):
    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            predictions = (outputs > 0.5).float()
            correct = (predictions == masks).float()
            accuracy = correct.sum() / correct.numel()
            total_correct += accuracy.item()

    avg_accuracy = total_correct / len(val_loader)
    print(f'Validation Accuracy: {avg_accuracy:.4f}')

    return avg_accuracy

def test_model_with_best(model, loader, save_dir, model_weights_path):
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            # 假设选择通道 index 为 1 作为牙齿的通道
            # teeth_channel = 1
            # segmentation_mask = (outputs.argmax(1) == teeth_channel).float()

            # Apply threshold if needed
            segmentation_mask = (outputs > 0).float()

            # Save the predictions with the same name and format as the original image
            img_name = test_dataset.image_paths[i]
            save_path = os.path.join(save_dir, img_name)
            print(save_path)

            # Convert predictions to PIL image without resizing
            predictions_pil = transforms.ToPILImage()(segmentation_mask.squeeze().cpu())

            # Save the PIL image
            predictions_pil.save(save_path)

def plot_training_curves(train_loss_list, train_accuracy_list, val_accuracy_list, num_epochs):
    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, label='Training Accuracy')
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()

    plt.show()



if __name__ == '__main__':
    # 数据目录
    data_dir = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_train\train'
    data_dir2 = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_image'

    # 数据转换
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # 例如，使用高斯滤波
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor()
    ])

    # 训练集_有标签
    labelled_train_image_folder = os.path.join(data_dir, 'labelled', 'image')
    labelled_train_mask_folder = os.path.join(data_dir, 'labelled', 'label')
    labelled_train_dataset = ToothDataset(labelled_train_image_folder, mask_folder=labelled_train_mask_folder,
                                          transform=transform)

    # 训练集_没有标签
    unlabelled_image_folder = os.path.join(data_dir, 'unlabelled', 'image')
    unlabelled_train_dataset = ToothDataset(unlabelled_image_folder, transform=transform)

    # 测试集
    test_image_folder = os.path.join(data_dir2, 'image')
    test_dataset = ToothDataset(test_image_folder, transform=transform)

    # 将有标签数据划分训练集和验证集
    train_size = int(0.8 * len(labelled_train_dataset))
    val_size = len(labelled_train_dataset) - train_size
    train_dataset, val_dataset = random_split(labelled_train_dataset, [train_size, val_size])

    # 创建 FCN 模型
    # model = FCN(1, 1)  # 根据你的任务设置输出通道数和类别数
    # model = ResUNet(1,1)
    # model = UNet2D(1,1)
    model = myUNet2D(1,1)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # DataLoader for testing
    unlabelled_train_loader = DataLoader(unlabelled_train_dataset, batch_size=1)

    # DataLoader for validation
    val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

    # 使用保存的最佳模型进行测试
    best_model_weights_path = 'best_model_fcn2.pth'
    test_save_dir = r"D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_train\train\unlabelled\label"
    test_save_dir2 = r"D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_image\infers_fcn2"
    # test_model_with_best(model, unlabelled_train_loader, test_save_dir, best_model_weights_path)
    test_model_with_best(model, test_loader, test_save_dir2, best_model_weights_path)

