from torch.utils.data import Dataset
import os
from PIL import Image
class MyData(Dataset):
    def __init__(self,root_dir,img_dir,label_dir):
        self.root_dir=root_dir
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.img_path=os.path.join(self.root_dir,self.img_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)

    def __getitem__(self, item):
        img_list=os.listdir(self.img_path)
        label_list=os.listdir(self.label_path)
        img_listpath=os.path.join(self.img_path,img_list[item])
        label_listpath = os.path.join(self.label_path, label_list[item])
        img=Image.open(img_listpath)
        label = open(label_listpath,"r",encoding="utf-8")
        return img,label

    def __len__(self):
        return len(self.img_path)
Ant = MyData('K:\\BaiduNetdiskDownload\\Dataset\\train','ants_image','ants_label')
img,label = Ant[7]
img.show()
print(label.read())