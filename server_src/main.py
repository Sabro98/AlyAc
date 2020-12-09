from flask import jsonify, Flask, escape, request
import json
import urllib
from urllib.request import Request, URLopener
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
import torch
from time import sleep
import os
import threading
import random

ERROR_MESSAGE = '네트워크 접속에 문제가 발생하였습니다. 잠시 후 다시 시도해주세요.'

app = Flask(__name__)


data_dir = './'
batch_size = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:0")

result = '예측을 위해 사진을 입력해주세요!'
flag = False

@app.route('/detail', methods=['POST'])
def detail():
    body = request.json
    pill_type = class_dict_inv[body['userRequest']['utterance']]

    info = readFile('./pill_info/{}.txt'.format(pill_type))
    link = readFile('./pill_info/{}_link.txt'.format(pill_type))
    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "basicCard": {
                        "title": class_dict[pill_type],
                        "description": info,
                        "thumbnail": {
                            "imageUrl": link.split('\n')[0].rstrip(),
                            "link": {
                                "web":  link.split('\n')[0].rstrip()
                            }
                        },
                        "buttons": [
                            {
                                "action": "webLink",
                                "label": "더 알아보기",
                                "webLinkUrl": link.split('\n')[1].rstrip()
                            }
                        ]
                    }
                }
            ]
        }
    }

    return jsonify(res)

def visualize_model(key):
    model = model_ft
    model.eval()

    data_transforms = {
    key: transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    transforms.Resize(448)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in [key]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4) for x in [key]}


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[key]):
            inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            accs, preds = torch.topk(outputs, 3)
            #
            # print(preds[0])

            result = []
            acc = []
            for i in range(3):
                if(accs[0][i].item() >= 0.1):
                    acc.append(int(accs[0][i].item() * 100))
                    result.append(class_names[preds[0][i]])
    os.remove('./{}/test/image.jpg'.format(key))
    os.rmdir('./{}/test'.format(key))
    os.rmdir('./{}'.format(key))
    return result, acc

def readFile(fileName):
    resultStr = ''
    file = open(fileName, encoding='utf-8')
    while True:
        info = file.readline()
        if not info:
            break
        resultStr += info
    file.close()
    return resultStr

def getResultJson(input_result):
    results, accs = input_result
    inf_arr = []
    link_arr = []
    for result in results:
        inf_arr.append(readFile('./pill_info/{}.txt'.format(result)))
        link_arr.append(readFile('./pill_info/{}_link.txt'.format(result)))

    if len(results) == 3:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel":{
                            "type": "basicCard",
                            "items":[
                                {
                                    "title": class_dict[results[0]] + ' (정확도 {}%)'.format(accs[0]),
                                    "description": inf_arr[0],
                                    "thumbnail": {
                                        "imageUrl": link_arr[0].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[0].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[0]]
                                        }
                                    ]
                                },
                                {
                                    "title": class_dict[results[1]] + ' (정확도 {}%)'.format(accs[1]),
                                    "description": inf_arr[1],
                                    "thumbnail": {
                                        "imageUrl": link_arr[1].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[1].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[1]]
                                        }
                                    ]
                                },
                                {
                                    "title": class_dict[results[2]] + ' (정확도 {}%)'.format(accs[2]),
                                    "description": inf_arr[2],
                                    "thumbnail": {
                                        "imageUrl": link_arr[2].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[2].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[2]]
                                        }
                                    ]
                                },
                                {
                                    "title": '제가 찾는 약이 아니에요 ㅠㅠ',
                                    "description": '직접 검색으로도 찾을 수 있습니다!',
                                    "thumbnail": {
                                        "imageUrl" : 'https://creazilla-store.fra1.digitaloceanspaces.com/emojis/56088/crying-face-emoji-clipart-md.png'
                                    },
                                    "buttons": [
                                        {
                                            "action": "webLink",
                                            "label": "직접 찾기",
                                            "webLinkUrl": 'http://www.health.kr/searchIdentity/search.asp'
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }
    elif len(results) == 2:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel":{
                            "type": "basicCard",
                            "items":[
                                {
                                    "title": class_dict[results[0]] + ' (정확도 {}%)'.format(accs[0]),
                                    "description": inf_arr[0],
                                    "thumbnail": {
                                        "imageUrl": link_arr[0].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[0].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[0]]
                                        }
                                    ]
                                },
                                {
                                    "title": class_dict[results[1]] + ' (정확도 {}%)'.format(accs[1]),
                                    "description": inf_arr[1],
                                    "thumbnail": {
                                        "imageUrl": link_arr[1].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[1].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[1]]
                                        }
                                    ]
                                },
                                {
                                    "title": '제가 찾는 약이 아니에요 ㅠㅠ',
                                    "description": '직접 검색으로도 찾을 수 있습니다!',
                                    "thumbnail": {
                                        "imageUrl": 'https://creazilla-store.fra1.digitaloceanspaces.com/emojis/56088/crying-face-emoji-clipart-md.png'
                                    },
                                    "buttons": [
                                        {
                                            "action": "webLink",
                                            "label": "직접 찾기",
                                            "webLinkUrl": 'http://www.health.kr/searchIdentity/search.asp'
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }
    elif len(results) == 1:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel": {
                            "type": "basicCard",
                            "items": [
                                {
                                    "title": class_dict[results[0]] + ' (정확도 {}%)'.format(accs[0]),
                                    "description": inf_arr[0],
                                    "thumbnail": {
                                        "imageUrl": link_arr[0].split('\n')[0].rstrip(),
                                        "link": {
                                            "web":  link_arr[0].split('\n')[0].rstrip()
                                        }
                                    },
                                    "buttons": [
                                        {
                                            "action": "message",
                                            "label": "이 약이 맞아요!",
                                            "messageText": class_dict[results[0]]
                                        }
                                    ]
                                },
                                {
                                    "title": '제가 찾는 약이 아니에요 ㅠㅠ',
                                    "description": '직접 검색으로도 찾을 수 있습니다!',
                                    "thumbnail": {
                                        "imageUrl": 'https://creazilla-store.fra1.digitaloceanspaces.com/emojis/56088/crying-face-emoji-clipart-md.png'
                                    },
                                    "buttons": [
                                        {
                                            "action": "webLink",
                                            "label": "직접 찾기",
                                            "webLinkUrl": 'http://www.health.kr/searchIdentity/search.asp'
                                        }
                                    ]
                                }
                            ]
                        }
                    }

                ]
            }
        }
    else:
        res = {
            "version": "2.0",
            "template":{
                "outputs":[
                    {
                        "basicCard":{
                            "title": "에측에 실패하였습니다...",
                            "description": "더욱 자세한 사진을 입력해주세요!",
                            "thumbnail": {
                                "imageUrl": 'https://creazilla-store.fra1.digitaloceanspaces.com/emojis/56088/crying-face-emoji-clipart-md.png'
                            },
                            "buttons": [
                                {
                                    "action": "weblink",
                                    "label": "그냥 직접 찾기!",
                                    "webLinkUrl": 'http://www.health.kr/searchIdentity/search.asp'
                                }
                            ]
                        }
                    }
                ]
            }
        }
    return res

flag = [False] * 100001
fail_res = [''] * 100001
def timer(key):
    global flag
    sleep(3.5)
    flag[key] = True
    global fail_res
    fail_res[key] = {
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text": "사진의 크기가 너무 크거나 네트워크 상태가 좋지 않습니다.\n 사진의 크기를 줄여주세요!!(도움말 참고)\n"
                    }
                }
            ]
        }
    }

@app.route('/search', methods=['POST'])
def search():
    key = random.randint(1, 100000)
    if flag[key]:
        flag[key] = False

    th = threading.Thread(target=timer, args=(key,))
    th.start()
    body = request.json
    img_url = body['action']['detailParams']['secureimage']['origin']
    img_url = img_url.replace(')', '(').replace(',', '(').split('(')[1]

    folder_name = './{}/test/'.format(key)
    os.makedirs(folder_name, exist_ok=True)
    # download = threading.Thread(target=download_img, args=(img_url,))
    # download.start()
    start = time.time()
    urllib.request.urlretrieve(img_url, folder_name+'image.jpg')
    end = time.time()
    print(end - start,'초(img)')
    print('Timeout:', flag[key])
    return jsonify(fail_res[key] if flag[key] else getResultJson(visualize_model(str(key))))


# 메인 함수
if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)

    model_ft = model_ft.to(device)
    class_names = []
    class_names_kr = []
    class_dict = {}
    class_dict_inv = {}
    labels = open('./model/labels.txt', 'r', encoding='utf-8')
    while True:
        name = labels.readline()
        if not name:
            break
        name = name.split(' ')
        class_names.append(name[0].rstrip())
        class_names_kr.append(name[1].rstrip())
        class_dict[name[0]] = name[1].rstrip()
        class_dict_inv[name[1].rstrip()] = name[0]
    labels.close()

    # from pprint import pprint as pp
    # pp(class_dict)

    path = 'model'+'.pth'
    model_ft.load_state_dict(torch.load('./model/'+path, map_location='cuda:0'))
    model_ft = nn.Sequential(model_ft, nn.Softmax())
    app.run(host='0.0.0.0', port=5000, threaded=True)

