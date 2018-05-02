# -*- coding: utf-8 -*-
"""
    用户使用界面来运用算法进行命名实体识别。
[1] 可以导入文本或者输入句子, 进行实体识别并给出显示结果
[2] 显示操作日志功能
"""
import sys
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QDesktopWidget,
                             QApplication, QAction, QHBoxLayout, QWidget, QGroupBox,
                             QVBoxLayout, QPushButton, QFormLayout, QLabel, QGridLayout,
                             QLineEdit, QFileDialog)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QRect
import codecs


import main
import threading
from multiprocessing.pool import ThreadPool


class MainEdit(QWidget):
    """自定义控件"""
    def __init__(self):
        super(MainEdit, self).__init__()

        self.initUI()

    def initUI(self):
        # 添加两个Label和TextEdit
        self.content = QLabel("Content")
        self.contentTextEdit = QTextEdit()
        self.result = QLabel("result")
        self.resultTextDdit = QTextEdit()

        # 创建一个网格布局
        grid = QGridLayout()
        # 设置组件之间的差距
        grid.setSpacing(50)

        grid.addWidget(self.content, 1, 0)
        grid.addWidget(self.contentTextEdit, 1, 1, 4, 1)

        grid.addWidget(self.result, 5, 0)
        grid.addWidget(self.resultTextDdit, 5, 1, 4, 1)

        # 设置布局
        self.setLayout(grid)

        # 设置窗口的位置和大小
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Review')
        self.show()


class MainWindow(QMainWindow):
    """
        先创建一个主窗口，添加菜单栏和状态栏，并添加相应事件；然后创建一
    个自定义的控件，使用网格布局加上相应的label和textedit。
    """
    def __init__(self):
        super(MainWindow, self).__init__()

        self.initUI()

    def initUI(self):
        # 添加状态栏
        statusBar = self.statusBar()

        # 添加MenuBar和menu
        menuBar = self.menuBar()
        operateMenu = menuBar.addMenu('&Operate')

        # 添加导入文本菜单项并创建导入文本事件
        importFileAction = QAction(QIcon('importFile.jpg'), 'Import File', self)
        importFileAction.setShortcut('Ctrl+F')
        importFileAction.setStatusTip('Import File')
        importFileAction.triggered.connect(self.importFile)

        operateMenu.addAction(importFileAction)

        # 添加实体识别菜单项并创建实体识别事件
        nerAction = QAction(QIcon('ner.jpg'), 'NER', self)
        nerAction.setShortcut('Ctrl+N')
        nerAction.setStatusTip('Name Entity Recognition')
        nerAction.triggered.connect(self.ner)

        operateMenu.addAction(nerAction)

        # 添加退出菜单项并创建退出事件
        exitAction = QAction(QIcon('exit.jpg'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit Application')
        exitAction.triggered.connect(self.close)

        operateMenu.addAction(exitAction)

        # 添加自定义的QWidget并居中
        self.textEdit = MainEdit()
        self.setCentralWidget(self.textEdit)

        # 设置窗口位置和大小
        self.resize(1400, 800)
        self.center()
        self.setWindowTitle('Main Window')
        self.show()

    def center(self):
        """使窗口居中"""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

    def importFile(self):
        """导入文件(支持UTF-8)"""
        # print("import file")
        fname = QFileDialog.getOpenFileName(self, 'Import File', '/home')
        if fname[0]:
            self.textEdit.contentTextEdit.setText("")
            with codecs.open(fname[0], 'r', 'utf-8') as f:
                data = f.read()
                self.textEdit.contentTextEdit.setText(data)

    def ner(self):
        """读入文本， 进行实体识别，并给出结果"""
        # 获取content TextEdit中的文本
        data = self.textEdit.contentTextEdit.toPlainText()
        # 一行行的读取并调用算法处理，将结果显示在result edit中
        # 开启线程调用算法进行实体识别
        # print('thread %s started.' % threading.current_thread().name)
        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(main.ner_text, (data, ))
        return_val = async_result.get()   # 得到返回值
        print("resut data", return_val)
        # self.textEdit.resultTextDdit.setText("")
        restr = ""
        for result in return_val:
            print(str(result))
            restr += str(result) + "\n"
        # print("-"*100)
        # print("restr: ", restr)
        self.textEdit.resultTextDdit.setText(restr)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
