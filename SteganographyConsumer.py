#!/usr/bin/env python3.4
## TODO: REMOVE HEADER KENT

import sys,re
from PySide.QtCore import *
from PySide.QtGui import *

from SteganographyGUI import *
from Steganography import *

from functools import partial
from scipy.misc import imread,imsave,imshow
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
        self.gv_obs = [self.viewPayload1,self.viewCarrier1,self.viewCarrier2,self.viewPayload2]

        #self.carrier_labels = [self.lblPayloadFound,self.lblCarrierEmpty]
        self.dragStartPosition = 0 # doesnt matter
        self.init_consumer()
        self.chkApplyCompression.clicked.connect(self.compressionClicked)
        self.btnSave.clicked.connect(self.saveInterface)
        self.slideCompression.valueChanged.connect(self.slideCompressionChanged)
        self.chkOverride.clicked.connect(self.check_can_save)
        self.btnClean.clicked.connect(self.clean_payload)
        self.btnExtract.clicked.connect(self.extract_view_payload)
        self.extract_filename = None

        #self.load_img("./color.png",0)
        #self.load_img("./medium.png",0)
        #self.load_img("./larger_test.jpg",0)
        #self.load_img("./nature.jpg",0)
        
    def check_can_save(self):
        if self.pc_obs[0] is None or self.pc_obs[1] is None:
            self.btnSave.setEnabled(False)
            return
        if(len(self.pc_obs[0].content) < np.size(self.pc_obs[1].img)//3\
               and (self.lblPayloadFound.text() == "" or \
                        self.chkOverride.isChecked())):
            self.btnSave.setEnabled(True)
        else:
            self.btnSave.setEnabled(False)

    def init_consumer(self):
        idx = 0
        for gv_ob in self.gv_obs:
            self.init_gvob(gv_ob,idx)
            idx += 1

    def init_gvob(self,gv_ob,index):


        ## MAKE AN INITIAL SCENE
        scene = QGraphicsScene()
        gv_ob.setScene(scene)
        if index == 3:
            return

        ## DRAG AND DROP FUNCTIONALITY
        gv_ob.mousePressEvent = self.my_mousePressEvent
        gv_ob.mouseMoveEvent = partial(self.my_mouseMoveEvent,index=index)
        gv_ob.showEvent = partial(self.my_showEvent,index=index)
        gv_ob.dragEnterEvent = self.my_dragEnterEvent
        gv_ob.dragMoveEvent = self.my_dragMoveEvent
        gv_ob.dropEvent = partial(self.my_dropEvent,index=index)
        gv_ob.setAcceptDrops(True)



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

        self.check_can_save()

    def slideCompressionChanged(self):
        compLevel = self.slideCompression.value()
        self.txtCompression.setText(str(compLevel))
        if self.pc_obs[0] is not None:
            self.pc_obs[0] = Payload(self.pc_obs[0].img,self.slideCompression.value())
            self.txtPayloadSize.setText(str(len(self.pc_obs[0].content)))        
        self.check_can_save()

    def my_showEvent(self,event,index):
        self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)

    def load_img(self,fn,index,in_img=None):
        ## READ FROM FILE IF NO IMAGE
        if in_img is None:
            try:
                mat = re.match(r"file:\/\/(?P<fn>.*\.[A-Za-z]+).*",fn)
                if mat is not None:
                    fn = mat.group(1)
                    in_img = imread(fn)
                    #my_buffer = request.urlopen(fn).read()
                    #img = Image.open(BytesIO(my_buffer))
                else:
                    in_img = imread(fn)
                img = Image.fromarray(in_img)
            except:
                return
        ## else INITIALIZE IMAGE VARIABLES
        else:
            img = Image.fromarray(imread(fn))
            
        ## MAKE A SCENE
        scene = QGraphicsScene()
        self.gv_obs[index].setScene(scene)

        ## LOAD IMAGE
        imgQ = ImageQt(img)
        pixMap = QPixmap.fromImage(imgQ)
        self.gv_obs[index].scene().addPixmap(pixMap)
        self.gv_obs[index].show()

        if index == 0:
            if self.slideCompression.isEnabled():
                self.pc_obs[index] = Payload(in_img,self.slideCompression.value())
            else:
                self.pc_obs[index] = Payload(in_img,-1)

            self.txtPayloadSize.setText(str(len(self.pc_obs[index].content)))
        else:
            self.pc_obs[index] = Carrier(in_img)
            self.txtCarrierSize.setText(str(np.size(self.pc_obs[index].img)//3))

            if self.pc_obs[index].payloadExists() and index == 1:
                self.lblPayloadFound.setText(">>>> Payload Found <<<<")
                self.chkOverride.setEnabled(True)
            elif index == 1:
                self.lblPayloadFound.setText("")
                self.chkOverride.setEnabled(False)
                self.chkOverride.setChecked(False)

            elif self.pc_obs[index].payloadExists() and index == 2:
                self.lblCarrierEmpty.setText("")
                self.btnClean.setEnabled(True)
                self.btnExtract.setEnabled(True)
            elif index == 2:
                self.lblCarrierEmpty.setText(">>>> Carrier Empty <<<<")
                self.btnClean.setEnabled(False)
                self.btnExtract.setEnabled(False)

            if index == 2:
                self.gv_obs[3].scene().clear()
                self.extract_filename = fn

        self.check_can_save()

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
        if index == 0:
            self.chkApplyCompression.setChecked(False)
            self.slideCompression.setSliderPosition(0)
            self.slideCompression.setEnabled(False)
            self.txtCompression.setEnabled(False)
            self.lblLevel.setEnabled(False)


        if (event.mimeData().hasImage()):
            npimg = self.convert_qimage_numpy(event.mimeData().imageData().toImage())
            self.load_img(None,index,npimg)
            self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)
        elif(event.mimeData().hasText()):
            self.load_img(event.mimeData().text(),index)
            self.gv_obs[index].fitInView(self.gv_obs[index].scene().sceneRect(),Qt.KeepAspectRatio)
        self.check_can_save()

    def my_dragMoveEvent(self,event):
        event.accept()

    def convert_qimage_numpy(self,incomingImage):
        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        arr = np.array(ptr).reshape(height, width, -1)  #  Copies the data
        return Image.fromarray(arr)

    def saveEmbeddedFile(self,fn):
        if self.btnSave.isEnabled() or self.lblPayloadFound.text() == "":
            self.pc_obs[1].clean()
            img = self.pc_obs[1].embedPayload(self.pc_obs[0])
        else:
            img = self.pc_obs[1].img
        imsave(fn,img)

    def saveInterface(self):
        """
        Obtain a file name from a file dialog, \
        and pass it on to the loading method.    
        """
        filePath, _ = QFileDialog.getSaveFileName(self, caption='Target File name and location')

        if not filePath:
            return

        self.saveEmbeddedFile(filePath)


    def clean_payload(self):
        self.btnClean.setEnabled(False)
        self.btnExtract.setEnabled(False)
        imsave(self.extract_filename,self.pc_obs[2].clean())

    def extract_view_payload(self):

        extracted = self.pc_obs[2].extractPayload()
        img = Image.fromarray(extracted.img)

        ## MAKE A SCENE
        scene = QGraphicsScene()
        self.gv_obs[3].setScene(scene)

        ## LOAD IMAGE
        imgQ = ImageQt(img)
        pixMap = QPixmap.fromImage(imgQ)
        self.gv_obs[3].scene().addPixmap(pixMap)
        self.gv_obs[3].show()
        self.btnExtract.setEnabled(False)
        self.gv_obs[3].fitInView(self.gv_obs[3].scene().sceneRect(),Qt.KeepAspectRatio)



if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = Consumer()

    currentForm.show()
    currentApp.exec_()


