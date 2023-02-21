"""

"""
import datetime
import os
import time
import  resnet
import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from dataProcess import  Process
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from model import CNNNet
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import torchvision.transforms as transforms
#系统参数
config = {
    'batchsize':2,
    'train_data_path':r'.\data\train',
    'val_data_path': r'.\data\val',
    'num_workers':1,
    'num_class':8,
    'lr':1e-3,
    'step_size':1,
    'gamma':0.98,
    'Epoch':50,
    'mode':'train',
    'model_save_name':'',
    'predict_mode':'pic_dir'

}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_save_path=''
model = torch.load('./model/model-2022-06-05 11-41-05/myCNNNet.pth')

#训练过程
def run(config):

    #数据处理
    train_dataset  = Process(data_path=config['train_data_path'],num_class=config['num_class'],img_size=224)
    TrainLoader = DataLoader(train_dataset,batch_size=config['batchsize'],
                             shuffle=True,num_workers=config['num_workers'])

    val_dataset  = Process(data_path=config['val_data_path'],num_class=config['num_class'],img_size=224)
    ValLoader = DataLoader(val_dataset,batch_size=config['batchsize'],
                           shuffle=True,num_workers=config['num_workers'])
    mapping = train_dataset.mapping
    demapping = {v:k for k,v in mapping.items()}

    # model  = CNNNet(num_class=config['num_class'])
    model=resnet.ResNet18()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)




    optimizer = optim.Adam(model.parameters(),lr=config['lr'])
    scheduler  = lr_scheduler.StepLR(optimizer,step_size=config['step_size'],
                        gamma=config['gamma'])
    lossFunc = torch.nn.CrossEntropyLoss()

    for epoch in range(config['Epoch']):
        running_loss = None
        pbar = tqdm(enumerate(TrainLoader),total = len(TrainLoader))
        for idx , (img,label) in pbar:
            img,label = img.to(device).float(),label.to(device).long()
            model.train()
            optimizer.zero_grad()
            pred = model(img)
            loss = lossFunc(pred,label)
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss=running_loss*0.99+loss.item()*0.01
            loss.backward()
            optimizer.step()
            description = f'epoch {epoch} loss:{running_loss}'
            pbar.set_description(description)
        scheduler.step()

        if epoch%2==0:
            print("-------------Val run--------------------")
            pbar = tqdm(enumerate(ValLoader),total = len(ValLoader))
            cls_pre_=[]
            cls_label_=[]
            for idx, (img, label) in pbar:
                img, label = img.to(device).float(), label.to(device).long()
                model.eval()
                pred = model(img)
                cls_pre = torch.argmax(pred,dim=1).cpu().detach().numpy().tolist()
                cls_label = label.cpu().detach().numpy().tolist()
                cls_pre_ += [demapping[cls] for cls in cls_pre]
                cls_label_ += [demapping[cls] for cls in cls_label]
            print(classification_report(cls_pre_,cls_label_))
            print("-----------------Val END----------------------")

    model_save_path = './model/model-' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')).replace(':', '-')
    print(model_save_path)
    os.mkdir(model_save_path)
    torch.save(model, model_save_path+'/myCNNNet.pth')

#解决在cv2中显示不了中文的问题
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "STSONG.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


#模型预测类型,返回中文类别名
def modelpredict(img):

    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = ((img / 255.0) - mean) / std
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]
    img = img.copy().transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, device=device)

    img = img.type(torch.cuda.FloatTensor)

    model.to(device)
    model.eval()

    mapping = {
        '猫': 0,
        '狗': 1,
        '熊猫': 2,
        '猪': 3,
        '男人': 4,
        '女人': 5,
        '皮卡丘': 6,
        '篮球':7
    }
    de_mapping = {v: k for k, v in mapping.items()}

    with torch.no_grad():
        model.eval()
        # print(img.shape)
        pred = model(img)
        cls_pre = torch.argmax(pred).cpu().detach().numpy().tolist()
        cls_pre_ = [de_mapping[cls_pre]]
    return cls_pre_
#实现图片类别预测或者预测视频中的物体类别
def predict(predict_mode):

    video_save_path = './predictImg/predict/predict.mp4'
    video_path = 0
    true_num=0
    picture_num=0
    if predict_mode == 'pic_dir':
        # for i in range(1, 20):
        #     img_path = './predictImg/img_' + str(i) + '.png'
        #     img = cv2.imread(img_path, 1)
        #     plt.imshow(img)
        #     if img.shape[2] != 3:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        #     mean = np.array([0.485, 0.456, 0.406])
        #     std = np.array([0.229, 0.224, 0.225])
        #     img = ((img / 255.0) - mean) / std
        #     img = cv2.resize(img, (224, 224))
        #     img = img[:, :, ::-1]
        #     img = img.copy().transpose(2, 0, 1)
        #     img = np.expand_dims(img, axis=0)
        #     img = torch.tensor(img, device=device)
        #
        #     img = img.type(torch.cuda.FloatTensor)
        #
        #     model.to(device)
        #     model.eval()
        #
        #     mapping = {
        #         '猫': 0,
        #         '狗': 1,
        #         '熊猫': 2,
        #         '猪猪': 3,
        #         '男人': 4,
        #         '女人': 5,
        #         '皮卡丘': 6
        #     }
        #     de_mapping = {v: k for k, v in mapping.items()}
        #
        #     with torch.no_grad():
        #         model.eval()
        #         # print(img.shape)
        #         pred = model(img)
        #         cls_pre = torch.argmax(pred).cpu().detach().numpy().tolist()
        #         cls_pre_ = [de_mapping[cls_pre]]
        #         print("the {}th picture is the class of : {}".format(i, cls_pre_))
        #
        #         plt.title("第 {}张图片的类别是 : {}".format(i, cls_pre_))
        #         plt.show()
        for filename in os.listdir(r"./predictImg"):  # listdir的参数是文件夹的路径
            if '.jpg' in filename:
                img_path = r"./predictImg/"+filename
                picture_num=picture_num+1
                img = cv2.imread(img_path, 1)
                plt.imshow(img)
                if img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = ((img / 255.0) - mean) / std
                img = cv2.resize(img, (224, 224))
                img = img[:, :, ::-1]
                img = img.copy().transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)
                img = torch.tensor(img, device=device)

                img = img.type(torch.cuda.FloatTensor)
                # img = img.unsqueeze(0)
                model.to(device)
                model.eval()

                mapping = {
                    'cat': 0,
                    'dog': 1,
                    'panda': 2,
                    'pig': 3,
                    'man': 4,
                    'woman': 5,
                    'picaqiu': 6,
                    'basketball':7
                }
                de_mapping = {v: k for k, v in mapping.items()}

                with torch.no_grad():
                    model.eval()
                    # print(img.shape)
                    pred = model(img)
                    cls_pre = torch.argmax(pred).cpu().detach().numpy().tolist()
                    cls_pre_ = de_mapping[cls_pre]
                    a = False
                    if img_path.split('Img/')[1].split('_')[0] == cls_pre_:
                        a = True
                        true_num = true_num + 1


                print("the {}  th picture is the class of : {} ".format(filename, cls_pre_),a)

                plt.title("第 {}张图片的类别是 : {} ".format(filename, cls_pre_))
                plt.show()
        print ("the total number of predict picture is ：{}, and the predict of true number is {}".format(picture_num,true_num))
        print("混合准确率：{}/{}= {}".format(true_num,picture_num,true_num/picture_num))
    if predict_mode == 'video':
        capture = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, 95.0, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            img = frame
            cls_pre_ = modelpredict(img)  # 模型预测，返回预测出来的类型

            fps = (fps + (1. / (time.time() - t1)+float(1e-8))) / 2
            print("fps= %.2f" % (fps) + ",类别是：{}".format(cls_pre_))
            # frame = cv2.putText(frame, "class is {}".format(cls_pre_), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2ImgAddText(frame, "这个是：{}".format(cls_pre_), 20, 20, (0, 255, 139), 40)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    config['mode']='train'
    config['predict_mode']='video'#video,pic_dir

    if config['mode']=='train':
        run(config)
    if config['mode']=='predict':

        predict(config['predict_mode'])










