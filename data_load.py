import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class ToothDataset(Dataset):
    def __init__(self, image_folder, mask_folder=None, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_paths = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_name = self.image_paths[index]
        img_path = os.path.join(self.image_folder, img_name)
        mask_path = os.path.join(self.mask_folder, img_name) if self.mask_folder else None

        image = Image.open(img_path).convert("L")  # "L"表示加载为灰度图像

        # 如果有mask，加载并进行二值化处理
        if mask_path:
            mask = Image.open(mask_path).convert("L")
            mask = transforms.ToTensor()(mask).to(dtype=torch.float32)
        else:
            # Create a placeholder mask filled with zeros
            mask = torch.zeros_like(transforms.ToTensor()(image), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, mask


if __name__ == '__main__':
    # 数据目录
    data_dir = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_train\train'
    data_dir2 = r'D:\pycharm_\python_code\homework\Mask_RCNN\Data\F_image'

    # 数据转换
    transform = transforms.Compose([transforms.ToTensor()])

    # 训练集
    labelled_train_image_folder = os.path.join(data_dir, 'labelled', 'image')
    labelled_train_mask_folder = os.path.join(data_dir, 'labelled', 'label')
    labelled_train_dataset = ToothDataset(labelled_train_image_folder, mask_folder=labelled_train_mask_folder, transform=transform)
    unlabelled_image_folder = os.path.join(data_dir, 'unlabelled', 'image')
    unlabelled_train_dataset = ToothDataset(unlabelled_image_folder, transform=transform)
    # 测试集
    test_image_folder = os.path.join(data_dir2, 'image')
    test_dataset = ToothDataset(test_image_folder, transform=transform)

    # 打印训练集和测试集的样本数
    print(f"Number of samples in train dataset: {len(labelled_train_dataset)+len(unlabelled_train_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")

    # 打印第一个训练样本的图像和mask
    sample_image, sample_mask = labelled_train_dataset[3]
    print("Sample Image (labelled_Train):")
    print(sample_image.shape)  # 打印图像张量的形状
    print("Sample Mask (Train):")
    print(sample_mask.shape)  # 打印mask张量的形状
    sample_image2,sample_mask2 = unlabelled_train_dataset[0]
    print("Sample Image (unlabelled_Train):")
    print(sample_image2.shape)  # 打印图像张量的形状
    print("Sample Mask (unlabelled_Train):")
    print(sample_mask2.shape)  # 打印图像张量的形状

    # 打印第一个测试样本的图像
    test_sample_image, _ = test_dataset[0]
    print("Sample Image (Test):")
    print(test_sample_image.shape)  # 打印测试图像张量的形状

