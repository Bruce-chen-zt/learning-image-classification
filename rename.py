import os

for filename in os.listdir(r"./predictImg"):              #listdir的参数是文件夹的路径
    if  '.jpg' in filename:
        print (r'./predictImg/'+ filename)  #此时的filename是文件夹中文件的名称

mapping = {
                '猫': 0,
                '狗': 1,
                '熊猫': 2,
                '猪猪': 3,
                '男人': 4,
                '女人': 5,
                '皮卡丘': 6,
                '篮球':7
            }
de_mapping = {v: k for k, v in mapping.items()}
# print(de_mapping)
# print(mapping)

