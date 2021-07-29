from PIL import Image
from torchvision import transforms
img_path = 'hymenoptera_data\\val\\ants\\8124241_36b290d372.jpg'#相对路径
img = Image.open(img_path)
tensor_writer = transforms.ToTensor()#创建具体的工具
img_tensor = tensor_writer(img)
print(img_tensor)