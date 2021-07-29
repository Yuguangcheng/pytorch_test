from torch.utils.data import Dataset
from PIL import Image
import os
class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir#获取相对地址
        self.label_dir=label_dir #对应标签地址
        self.path=os.path.join(self.root_dir,self.label_dir)#拼接两个地址
        self.img_path=os.listdir(self.path)#获取图片命名 以列表述方式

    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)#单个图片实际地址
        img=Image.open(img_item_path)
        label=self.label_dir
        return label,img

    def __len__(self):
        return len(self.img_path)

ants_dataset=Mydata('hymenoptera_data/train','ants')
label,img=ants_dataset.__getitem__(2)
img.show()
print(ants_dataset.__len__())