import os,sys,time,threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

#私有类
class _ImageListWidget(QListWidget):
    def __init__(self):
        super(_ImageListWidget, self).__init__()
        self.setFlow(QListView.Flow(1))  #QListView.Flow用于控制视图中的数据排列方向，0: 从左到右,1: 从上到下
        self.setIconSize(QSize(150,100))

    def add_image_items(self,image_paths=[]):  #添加图片进入右侧列表
        for img_path in image_paths:
            if os.path.isfile(img_path):  #os.path.isfile判断路径是否为文件
                img_name = os.path.basename(img_path)  #os.path.basename用于返回img_path的文件名，赋值给img_name
                item = QListWidgetItem(QIcon(img_path),img_name)  #QListWidgetItem用于创建拷贝，拷贝对象为img_path的图像文件，利用QIcon生成缩略图，命名为img_name
                self.addItem(item)  #每次for循环都向列表中添加创建的item


class ImageViewerDialog(QDialog,QThread):   
    def __init__(self,img_dir):                   
        super(QDialog, self).__init__()      
        self.setModal(False)   # 非模态窗体        
        self.list_widget = _ImageListWidget()  #调用私有类
        self.list_widget.setMinimumWidth(200)
        self.list_widget.setMaximumWidth(300) #限制控件的最大最小尺寸
        self.show_label = QLabel(self)
        self.show_label.setFixedSize(800,400)
        self.image_paths = []  #创建image_paths列表，列表中每一项存储爬取图片的路径
        self.currentImgIdx = 0
        self.currentImg = None
        self.setWindowIcon(QIcon('./UI/icon.jpg'))
        self.layout = QHBoxLayout(self)     # 水平布局
        self.layout.addWidget(self.show_label)
        self.layout.addWidget(self.list_widget)

        self.list_widget.itemSelectionChanged.connect(self.loadImage)   #当选中的缩略图状态发生变化时，链接装载并显示缩略图的槽函数

        try:   
            filenames = os.listdir(img_dir)   #获取图片包含拓展名的全名               
            img_paths=[]                                         
            for file in filenames:  #对每一个图片进行循环                           
                if file[-4:]==".png" or file[-4:]==".jpg": #若图片文件的最后四个字符，也即扩展名为.jpg或.png，则将对应的图片添加到缩略图列表中          
                    img_paths.append(os.path.join(img_dir,file))   #append在列表最后添加新的对象，对象为os.join函数拼接的  
            self.load_from_paths(img_paths) 
        except Exception as e:  #若文件夹路径无符合特征的图片，则print错误信息
            print("no img_dir{0}".format(img_dir),e)  #.format为格式化函数，将字符串中的{}替换，按照0,1,2...的顺序替换                  
    
        self.setWindowTitle("缩略图预览")
        self.resize(1200,600) 
        self.show()

    def load_from_paths(self,img_paths=[]):
        self.image_paths = img_paths
        self.list_widget.add_image_items(img_paths)

    def loadImage(self):
        self.currentImgIdx = self.list_widget.currentIndex().row()
        if self.currentImgIdx in range(len(self.image_paths)):
            self.currentImg = QPixmap(self.image_paths[self.currentImgIdx]).scaledToHeight(400)
            self.show_label.setPixmap(self.currentImg)
            self.show_label.setAlignment(Qt.AlignCenter) #左侧控件装载并显示图片的缩略图

