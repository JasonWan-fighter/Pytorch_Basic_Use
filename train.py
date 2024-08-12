"""完整的模型训练套路"""
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(".\\CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root=".\\CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 利用DataLoader 来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learing rate = 0.01
learing_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learing_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter(".\\logs_train")

for i in range(epoch):
    print("----第{}轮训练开始------".format(i + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
