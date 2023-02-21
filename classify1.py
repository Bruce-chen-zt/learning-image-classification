"""

"""

import os
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import torch

#系统参数

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device( 'cpu')

model_save_path=''
model = torch.load('./model/model-2022-06-16 16-11-37/myCNNNet.pth')

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

    img = img.type(torch.FloatTensor if device!='cuda:0' else torch.cuda.FloatTensor)

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
def predict( ):

    video_save_path = ''
    video_path = 0
    true_num=0
    picture_num=0

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

        fps = (fps + (1. / (time.time() - t1) + float(1e-8))) / 2
        print("fps= %.2f" % (fps) + ",类别是：{}".format(cls_pre_))
        # frame = cv2.putText(frame, "class is {}".format(cls_pre_), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2ImgAddText(frame, "欢迎使用：陈泽涛分类系统----这个是：{}".format(cls_pre_), 20, 20, (0, 122, 139), 20)
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



    predict()










