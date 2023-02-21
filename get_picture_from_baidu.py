# -*- coding:utf-8 -*-
import requests
import re, time, datetime
import os
import random
import urllib.parse
from PIL import Image  # 导入一个模块

imgDir = r"./data/train/wave/"
# 设置headers 为了防止反扒，设置多个headers
# chrome，firefox，Edge
headers = [
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive'
    },
    {
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive'
    },
    {
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19041',
        'Accept-Language': 'zh-CN',
        'Connection': 'keep-alive'
    }
]

picList = []  # 存储图片的空 List

keyword = input("请输入搜索的关键词：")
kw = urllib.parse.quote(keyword)  # 转码


# 获取 1000 张百度搜索出来的缩略图 list
def getPicList(kw, n):
    global picList
    weburl = r"https://image.baidu.com/search/acjson?tn=resultjson_com&logid=11601692320226504094&ipn=rj&ct=201326592&is=&fp=result&queryWord={kw}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word={kw}&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&force=&cg=girl&pn={n}&rn=30&gsm=1e&1611751343367=".format(
        kw=kw, n=n * 30)
    req = requests.get(url=weburl, headers=random.choice(headers))
    req.encoding = req.apparent_encoding  # 防止中文乱码
    webJSON = req.text
    imgurlReg = '"thumbURL":"(.*?)"'  # 正则
    picList = picList + re.findall(imgurlReg, webJSON, re.DOTALL | re.I)


for i in range(150):  # 循环数比较大，如果实际上没有这么多图，那么 picList 数据不会增加。
    getPicList(kw, i)
j =1
for item in picList:
    # 后缀名 和名字
    itemList = item.split(".")
    hz = ".jpg"
    label='wave_'
    picName = str(int(time.time() * 1000))  # 毫秒级时间戳
    # 请求图片
    imgReq = requests.get(url=item, headers=random.choice(headers))
    # 保存图片
    with open(imgDir + label+str(j) + hz, "wb") as f:
        f.write(imgReq.content)
    #  用 Image 模块打开图片
    im = Image.open(imgDir + label+str(j) + hz)
    # bili = im.width / im.height  # 获取宽高比例，根据宽高比例调整图片大小
    # newIm = None
    # # 调整图片的大小，最小的一边设置为 50
    # if bili >= 1:
    #     newIm = im.resize((round(bili * 50), 50))
    # else:
    #     newIm = im.resize((50, round(50 * im.height / im.width)))
    # # 截取图片中 50*50 的部分
    clip = im # 截取图片,crop 裁切
    clip.convert("RGB").save(imgDir + label+str(j) + hz)  # 保存截取的图片
    j=j+1
    print(picName + hz + " 处理完毕")
