import os,cv2
import numpy as np
class Process():
    def __init__(self,data_path,num_class,img_size = 224):
        self.data_path = data_path
        self.num_class = num_class
        self.img_size = img_size
        self.catNames = os.listdir(self.data_path+'/cat')
        self.dogNames = os.listdir(self.data_path+'/dog')
        self.pandaNames = os.listdir(self.data_path+'/panda')
        self.pigNames = os.listdir(self.data_path+'/pig')
        self.manNames = os.listdir(self.data_path+'/man')
        self.womanNames = os.listdir(self.data_path+'/woman')
        self.picaqiuNames = os.listdir(self.data_path+'/picaqiu')
        self.basketballNames = os.listdir(self.data_path+'/basketball')

        self.totalNames = self.catNames+self.dogNames+self.pandaNames+\
                          self.pigNames+self.manNames+self.womanNames+self.picaqiuNames+self.basketballNames
        np.random.seed(seed=666)
        np.random.shuffle(self.totalNames)
        self.mapping={
            'cat':0,
            'dog':1,
            'panda':2,
            'pig':3,
            'man':4,
            'woman':5,
            'picaqiu':6,
            'basketball':7

        }

    def __len__(self):
            return len(self.totalNames)
    def __getitem__(self,item):
            item = item%len(self.totalNames)
            imgName = self.totalNames[item]
            cls = imgName.split('_')[0]
            label  = self.mapping[cls]
            img = cv2.imread(os.path.join(self.data_path+'/{}'.format(cls),imgName))
            # if img.shape[2]!=3:
            #     img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            mean = np.array([0.485,0.456,0.406])
            std = np.array([0.229,0.224,0.225])
            img = ((img/255.0)-mean)/std
            img = cv2.resize(img,(self.img_size,self.img_size))
            img = img[:,:,::-1]
            img = img.copy().transpose(2,0,1)
            return img,label