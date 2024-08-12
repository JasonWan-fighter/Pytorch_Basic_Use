import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# 写标量
writer = SummaryWriter('logs')

# 写图像
# add_image和add_scalar的参数和用法十分相似，一个添加标量，一个添加图像，
# 只是添加的图像必须是tensor或numpy.array类型，要么用opencv读入图片，要么转换
image_path = ".\\dataset\\train\\ants\\0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)


for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
                                                      # 记得指明你的图像的格式
writer.add_image("test", img_array, 1, dataformats='HWC')
writer.close()


