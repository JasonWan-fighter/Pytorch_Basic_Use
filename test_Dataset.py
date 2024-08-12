from torch.utils.data import Dataset
from PIL import Image
import os

"""
apparently,
    这个Dataset类关键是给到所有要用的路径，
    然后可以根据路径，通过PIL库来获取对应图片
    继承Dataset的意义在于能够实现合并数据集等功能
----------------------------------------------
数据的组织形式，具有同一个标签的数据都放在同一个文件夹里，文件夹名即是标签

读取数据：自己写一个Data类并继承Dataset，
"""


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset  # 这个+号能直接相加，是继承Dataset的意义
print(ants_dataset[0])
