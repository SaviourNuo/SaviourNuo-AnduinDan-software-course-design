import sys
from PyQt5.QtWidgets import QApplication
import UI.GUI_SCD3 
import neural_networks.pytorch_classification.Test5_resnet as resModel
import GlobalVariable as glovar
sys.path.append(".") #将当前工作目录添加到寻找module的path里

if __name__ == "__main__":
    glovar._init() # 初始化工程共享变量表，用于共享“目前爬取到的数字”和“总爬取数量”，辅助完成爬虫和进度条的同步
    app = QApplication(sys.argv)  #传入所有参数
    mainWindow = UI.GUI_SCD3.AppMainWindow() #生成主窗体对象
    sys.exit(app.exec_()) #主线程退出