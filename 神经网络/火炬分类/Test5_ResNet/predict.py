import os
import json
import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from PyQt5.QtCore import QThread

#from neural_networks.pytorch_classification.Test5_resnet.model import resnet34

# 这个批量推理啊，要记得最后把图片分别写到对应类别的文件夹里去哈
def predict(imgPath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './neural_networks/pytorch_classification/text_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = torchvision.models.resnet50(num_classes=2).to(device)# resnet34(num_classes=2).to(device)# create model

    # load model weights
    weights_path = "./neural_networks/weights_list/resNet50_test7_SGD.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # load image
    #imgPath = "./network_spider/getImage2_test/{0}".format(keyword)
    #assert os.path.exists(imgPath), "file: '{}' dose not exist.".format(imgPath)
    if not os.path.exists(imgPath):
        print("invalid image path")
        return -2
    imgPathList = os.listdir(imgPath) # 取出图片列表

    # print(222)

    #set output path
    outputPath = "./neural_networks/classification_output/{0}".format(os.path.basename(imgPath))# 推理输出的根目录
    if not os.path.exists(outputPath):
        os.makedirs(outputPath) 
    for i in range(len(class_indict)):
        claPath = os.path.join(outputPath,class_indict[str(i)]) # 推理输出的根目录下，每个类别的目录
        if not os.path.exists(claPath):
            os.mkdir(claPath)
    
    # print(333)

    for imgName in (imgPathList):
        try: 
            imgOrg = Image.open(os.path.join(imgPath,imgName)).convert('RGB')
        except Exception as e:
            print("{0}:{1}".format(imgName,e))
            continue
        print(imgName)
        img = data_transform(imgOrg)
        img = torch.unsqueeze(img, dim=0)# expand batch dimension

        model.eval()# prediction
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu() # predict class
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        for i in range(len(class_indict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                    predict[i].numpy()))

        # 把推理后的图片存放到指定的文件夹里去
        try:
            imgOutPath = os.path.join(outputPath,class_indict[str(predict_cla)],imgName)
            imgOrg.save(imgOutPath)
        except Exception as e:
            print(imgOutPath,e)
            return -1
        
    # print(444)

    print("predict end")
    return 0

if __name__ == '__main__':
    # keywords = ["街道","足球","机械","果蔬","医药","动漫","招牌","海洋","宇宙","肖像"]
    keywords = ["non-text","text"]  # 对验证集集做推理，人眼来看看f1一直提不上去的原因
    for keyword in keywords:
        predict("./network_spider/getImage2_test/{0}".format(keyword))

class PredictThread(QThread):
    def __init__(self,imgPath):                             
        super(QThread,self).__init__()
        self.imgPath = imgPath   
        # print(self.imgPath)           
    def run(self): 
        #print(111)
        predict(self.imgPath)
        