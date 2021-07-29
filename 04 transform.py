from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
img_path = 'hymenoptera_data\\val\\ants\\8124241_36b290d372.jpg'#相对路径
img = Image.open(img_path)
tensor_writer = transforms.ToTensor()#创建具体的工具
img_tensor = tensor_writer(img)
writer = SummaryWriter("tsf_logs")
writer.add_image('tensor_img',img_tensor)
writer.close()