from datetime import datetime
import os,sys
import json
from tqdm import tqdm  # 进度条

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter, writer
sys.path.append(".") #将当前工作目录添加到找module的path里
#from neural_networks.pytorch_classification.Test5_resnet.model import resnet34



def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}   
    
    data_root = os.path.abspath(os.path.curdir)  # get data root path
    print(data_root)
    image_path = os.path.join(data_root,"neural_networks","data_set", "textDis_data")  # data set path
    
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'text':0, 'non-text':1}     # write dict into json file
    # photo_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in photo_list.items())
    # json_str = json.dumps(cla_dict, indent=4)
    # textClassDir = os.path.join(data_root,"neural_networks/pytorch_classification/text_class_indices.json")
    # with open(textClassDir, 'w') as json_file:
    #     json_file.write(json_str)

    batch_size = 4#8#32#128#32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    net = torchvision.models.resnet50(False)#  resnet50() #我自己写的resnet#torchvision里的resnet
    #torchvision.models.resnet34(True)#  resnet34()
    # load pretrain weights # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    in_channel = net.fc.in_features    # 根据要分的类数，构造FC层的维数
    net.fc = nn.Linear(in_channel, 2) # 二元分类问题
    model_weight_path = os.path.join(data_root,"neural_networks/weights_list/resNet50_test3_SGD(刚刚93).pth") # 导入预训练或已训练权重
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))   

    net.to(device)



    # Tensorboard 训练可视化
    images, labels = next(iter(train_loader))
    timeStamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(os.path.join(data_root,"neural_networks/pytorch_classification/textDis_Tensorboard_test",timeStamp),flush_secs=30)
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(net.to(device),images.to(device))

    loss_function = nn.CrossEntropyLoss() # 损失函数 用交叉熵损失
    params = [p for p in net.parameters() if p.requires_grad]
    #optimizer = optim.Adam(params,lr=0.00001,weight_decay=0.0002)    # 优化器，用adamSGD
    #optimizer = optim.Adam(params,lr=0.00001,weight_decay=0.0005)    # 优化器，用adamSGD
    #optimizer = optim.Adam(params, lr=0.0001)    # 优化器，用adamSGD
    optimizer = optim.SGD(params, lr=0.0001,momentum=0.9,weight_decay=0.0008)   
    #optim.SGD(params, lr=0.0001,momentum=0.9,weight_decay=0.0005)   
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=75,gamma=0.1,last_epoch=-1) #学习率衰减(要再不行就换余弦退火了)
                    #CosineAnnealingLR(optimizer=optimizer,T_max=10,eta_min=0,last_epoch=-1)
                    #

    epochs = 1000  #训练轮数
    best_acc = 0.0
    best_Fmeasure = 0.0
    save_path = os.path.join(data_root,'neural_networks/weights_list/resNet50_test7_SGD.pth')
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,unit = "img")
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # print statistics
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_1_label_1 = 0.0
        val_0_label_1 = 0.0
        val_1_label_0 = 0.0
        val_0_label_0 = 0.0
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, unit = "img")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                _ones  = torch.ones(len(val_labels)).to(device)
                #zeros = torch.zeros(len(val_labels)).to(device)
                val_1_label_1 += torch.logical_and(torch.eq(predict_y,val_labels.to(device)),torch.eq(predict_y,_ones)).sum().item()#.to("cpu").sum()#.cpu().sum()##.item()
                val_0_label_1 += torch.logical_and(torch.ne(predict_y,val_labels.to(device)),torch.ne(predict_y,_ones)).sum().item()#.to("cpu").sum()#.cpu().sum()##.item()
                val_1_label_0 += torch.logical_and(torch.ne(predict_y,val_labels.to(device)),torch.eq(predict_y,_ones)).sum().item()#.to("cpu").sum()#.cpu().sum()##.item()
                val_0_label_0 += torch.logical_and(torch.eq(predict_y,val_labels.to(device)),torch.ne(predict_y,_ones)).sum().item()#.to("cpu").sum()#.cpu().sum()##.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

        print("V1L1:{0}    V1L0:{1}    V0L1:{2}    V0L0:{3}".format(val_1_label_1,val_1_label_0,val_0_label_1,val_0_label_0))
        val_precision = val_1_label_1/(val_1_label_1+val_1_label_0)
        val_recall = val_1_label_1/(val_1_label_1+val_0_label_1)
        val_Fmeasure = 2*val_precision*val_recall/(val_precision+val_recall)
        val_acc = acc / val_num

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss/len(validate_loader),val_acc))

        writer.add_scalar("Evaluate/val_acc", val_acc, epoch)  # 画准确率
        writer.add_scalar("Evaluate/F-measure", val_Fmeasure, epoch)  # 画fmeasure
        writer.add_scalar("Evaluate/precision", val_precision, epoch)  # 画精确度
        writer.add_scalar("Evaluate/recall", val_recall, epoch)  # 画召回率
        writer.add_scalar("Loss/train_loss", running_loss/ train_steps, epoch) # 画损失函数
        writer.add_scalar("Loss/val_loss", val_loss/len(validate_loader), epoch) # 画损失函数
        writer.add_scalar("Parameters/learning_rate", optimizer.param_groups[0]["lr"], epoch)
       
        #scheduler.step() # 这一轮训练结束，衰减学习率

        #if val_acc > best_acc: #存权重表
        #    best_acc = val_acc
        if val_Fmeasure > best_Fmeasure and val_acc > 0.85 :
            best_Fmeasure = val_Fmeasure
            best_acc = max(best_acc,val_acc)
            torch.save(net.state_dict(), save_path)

        writer.close()

    print("best_acc = {0}; best_Fmeasure = {1}".format(best_acc,best_Fmeasure))
    print('Finished Training')
    

    return 0

if __name__ == '__main__':
    train()
