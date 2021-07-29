from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os
img_ant_list=os.listdir('dataset\\val\\ants')

writer = SummaryWriter("logs") #将事件文件存储到logs文件夹当中
'''y = x'''
for i in range(50):
    img_ant_title = img_ant_list[i]
    img_ant_path = os.path.join("dataset\\val\\ants", img_ant_title)
    img_ant_PIL = Image.open(img_ant_path)
    img_ant_Array = np.array(img_ant_PIL)
    writer.add_scalar('y = x', i, i)
    writer.add_scalar('y = 2x',2 * i,i)
    writer.add_scalar('y = 3x', 3 * i, i)
    writer.add_image('my_image_HWC', img_ant_Array ,1, dataformats='HWC')
''' tensorboard --logdir=logs --port=6007'''#打开事件位置和设置端口
writer.close()
