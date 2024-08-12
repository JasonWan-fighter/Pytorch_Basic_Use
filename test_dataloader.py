from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(".\\CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64,shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter(".\\logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data  # dataloader的作用就在于将数据进行打包分批
        writer.add_images("Epoch:{}".format(epoch), imgs, step)  # add_images会将分批好传进来的图片合成一张写入
        step = step + 1

writer.close()
print("finish")