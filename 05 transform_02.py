from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision  import transforms

img_path = 'hymenoptera_data\\val\\bees\\54736755_c057723f64.jpg'
img_PIL = Image.open(img_path)

#tensorboard
writer = SummaryWriter('logs')

#Totensor
tensor_tool = transforms.ToTensor()
img_Tensor = tensor_tool(img_PIL)
writer.add_image('Tensor_image',img_Tensor,1)

#Normalize 归一化
nor_tool = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_normal = nor_tool(img_Tensor)
writer.add_image('Normalize_image',img_normal,2)

#resize
print(img_PIL.size)
tensor_resize = transforms.Resize((512,512))
img_resize = tensor_resize(img_PIL)
img_resize = tensor_tool(img_resize)
writer.add_image('Resize_image',img_resize,3)
#compose
tensor_compose = transforms.Compose([transforms.Resize(512,),transforms.ToTensor()])
img_compose = tensor_compose(img_PIL)
writer.add_image('Compose_image',img_compose,5)
print(img_compose.size)

#Random crop 随机裁剪
tensor_Random = transforms.RandomCrop(200)
random_tool = transforms.Compose([tensor_Random,transforms.ToTensor()])
for i in range(10):
    img_random = random_tool(img_PIL)
    writer.add_image('Random_image', img_random, i)
writer.close()