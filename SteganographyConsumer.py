#!/usr/bin/env python3.4
## TODO: REMOVE HEADER KENT

import sys
from PySide.QtCore import *
from PySide.QtGui import *

from SteganographyGUI import *
from Steganography import *

from functools import partial
from scipy.misc import imread,imshow
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
from urllib import request
from io import BytesIO

class Consumer(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Consumer, self).__init__(parent)
        self.setupUi(self)
        self.pc_obs = [None,None,None,None]
        self.gv_obs = [self.viewPayload1,self.viewCarrier1,self.viewPayload2,self.viewCarrier2]


        print(self.viewPayload1.scene())

        self.dragStartPosition = 0 # doesnt matter
        self.init_consumer()
        self.chkApplyCompression.clicked.connect(self.compressionClicked)
        self.slideCompression.valueChanged.connect(self.slideCompressionChanged)

        #self.load_img("./color.png",0)
        self.load_img("./medium.png",0)
        #self.load_img("./larger_test.jpg",0)
        #self.load_img("./nature.jpg",0)
        
    def init_consumer(self):
        idx = 0
        for gv_ob in self.gv_obs:
            self.init_gvob(gv_ob,idx)
            idx += 1

    def init_gvob(self,gv_ob,index):

        ## DRAG AND DROP FUNCTIONALITY
        #print(dir(gv_ob))
        # print(dir(gv_ob))
        gv_ob.mousePressEvent = self.my_mousePressEvent
        gv_ob.mouseMoveEvent = partial(self.my_mouseMoveEvent,index=index)
        gv_ob.showEvent = partial(self.my_showEvent,index=index)
        gv_ob.dragEnterEvent = self.my_dragEnterEvent
        gv_ob.dragMoveEvent = self.my_dragMoveEvent
        gv_ob.dropEvent = partial(self.my_dropEvent,index=index)
        gv_ob.setAcceptDrops(True)

        ## MAKE A SCENE
        scene = QGraphicsScene()
        gv_ob.setScene(scene)


    def compressionClicked(self):
        if self.chkApplyCompression.isChecked():
            self.slideCompression.setEnabled(True)
            self.txtCompression.setEnabled(True)
            self.lblLevel.setEnabled(True)
            compLevel = self.slideCompression.value()
            self.txtCompression.setText(str(compLevel))
            if self.pc_obs[0] is not None:
                self.pc_obs[0] = Payload(self.pc_obs[0].img,compLevel)
                self.txtPayloadSize.setText(str(len(self.pc_obs[0].content)))        

        else:
            self.slideCompression.setEnabled(False)
            self.txtCompression.setEnabled(False)
            self.lblLevel.setEnabled(False)
            compLevel = self.slideCompression.value()
            self.txtCompression.setText(str(compLevel))
            if self.pc_obs[0] is not None:
                self.pc_obs[0] = Payload(self.pc_obs[0].img)
                self.txtPayloadSize.setText(str(len(self.pc_obs[0].content)))        


    def slideCompressionChanged(self):
        compLevel = self.slideCompression.value()
        self.txtCompression.setText(str(compLevel))
        if self.pc_obs[0] is not None:
            self.pc_obs[0] = Payload(self.pc_obs[0].img,self.slideCompression.value())
            self.txtPayloadSize.setText(str(len(self.pc_obs[0].content)))        

    def my_showEvent(self,event,index):
        self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)

    def load_img(self,fn,index,in_img=None):
        print(fn)
        if in_img is None:
            try:
                if "file://" == fn[0:7]:
                    my_buffer = request.urlopen(fn).read()
                    img = Image.open(BytesIO(my_buffer))
                else:
                    img = Image.fromarray(imread(fn))
                    #img = Image.open(fn)
            except:
                return
        else:
            img = Image.fromarray(in_img)

        imgQ = ImageQt(img)
        pixMap = QPixmap.fromImage(imgQ)
        self.gv_obs[index].scene().addPixmap(pixMap)
        self.gv_obs[index].show()

        try:
            npimg = np.asarray(img.convert("RGB"))
        except:
            ## TODO: clear scene
            return
        print("numpy made")
        print(npimg.shape)
        if index % 2 == 0:
            if self.slideCompression.isEnabled():
                self.pc_obs[index] = Payload(npimg,self.slideCompression.value())
            else:
                self.pc_obs[index] = Payload(npimg,-1)
            self.txtPayloadSize.setText(str(np.size(self.pc_obs[index].content)))
        else:
            self.pc_obs[index] = Carrier(npimg)
            self.txtCarrierSize.setText(str(np.size(self.pc_obs[index].img)))

    def my_mousePressEvent(self,my_QMouseEvent):
        if (my_QMouseEvent.button() == Qt.LeftButton):
            self.dragStartPosition = my_QMouseEvent.pos()
                
    def my_mouseMoveEvent(self,my_QMouseEvent,index):
        if (not my_QMouseEvent.buttons() and Qt.LeftButton):
            return
        if ((my_QMouseEvent.pos() - self.dragStartPosition).manhattanLength()
            < QApplication.startDragDistance()):
            return
        # we only drag if graphicsview has data
        #if (self.gv_obs[index]
        drag = QDrag(self)
        mimeData = QMimeData(self)
        if len(self.gv_obs[index].scene().items()) > 0:
            print(self.gv_obs[index].scene().items()[0].pixmap())
            mimeData.setImageData(self.gv_obs[index].scene().items()[0].pixmap())
            drag.setMimeData(mimeData)
            dropAction = drag.exec_()
             
    def my_mouseReleaseEvent(self, my_QMouseEvent):
        if (my_QMouseEvent.button() == Qt.LeftButton):
             dragEndPosition = my_QMouseEvent.pos()
        
    def my_dragEnterEvent(self,event):
        if (event.mimeData().hasText() or event.mimeData().hasImage()):
            event.acceptProposedAction()

    def my_dropEvent(self,event,index):
        self.gv_obs[index].scene().clear()
        if (event.mimeData().hasImage()):
            npimg = self.convert_qimage_numpy(event.mimeData().imageData())
            self.load_img(None,index,npimg)
            self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)
        elif(event.mimeData().hasText()):
            self.load_img(event.mimeData().text(),index)
            self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)
    def my_dragMoveEvent(self,event):
        event.accept()

    def convert_qimage_numpy(self,incomingImage):
        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, -1)  #  Copies the data
        return arr

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = Consumer()

    currentForm.show()
    currentApp.exec_()


