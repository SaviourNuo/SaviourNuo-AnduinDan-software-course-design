# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import GlobalVariable as glovar

#模态报错框生成，输入变量为需要在模态框中提示的错误文字
class ModalErrorText(QWidget):
    def __init__(self,errorText,width=300,height=200):
        super(ModalErrorText,self).__init__()
        self.resize(width, height)
        self.textLabel = QtWidgets.QLabel(self)
        self.textLabel.resize(width, 24)
        self.textLabel.move(50, 50)
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setText(errorText)

        
class ModalErrorTextDialog(QDialog,QThread):        
    def __init__(self,errorText="error",width=400,height=150):  
        super(QDialog, self).__init__()          
        self.setModal(True)  # 模态     
        self.modalErrorText = ModalErrorText(errorText)
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.modalErrorText)
        self.setWindowTitle("Error")
        self.setFixedSize(width,height)
        self.show()