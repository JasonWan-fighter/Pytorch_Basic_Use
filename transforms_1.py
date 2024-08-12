from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = '.\\pytorch.png'
img_PIL = Image.open(img_path)

# to tensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)

# normalize
trans_norm = transforms.Normalize([3, 1, 3, 5], [3, 3, 2, 1])
img_norm = trans_norm(img_tensor)

# resize
trans_resize = transforms.Resize((512, 512))
trans_compose = transforms.Compose([trans_resize, trans_totensor])
img_resized = trans_compose(img_PIL)
writer = SummaryWriter("logs")
writer.add_image("pytorch", img_resized, 2)


# RandomCrop
trans_RandomCropHW = transforms.RandomCrop((500, 1000))
trans_compose_2 = transforms.Compose([trans_RandomCropHW, trans_totensor])
for i in range(10):
    img_RC = trans_compose_2(img_PIL)
    writer.add_image("RCHW", img_RC, i)


writer.close()

print("finish")
