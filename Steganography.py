#!/usr/bin/env python3.4
import base64,zlib,re
from PIL import Image
import numpy as np
from timer import Timer ## TODO REMOVE THIS


def myfunc(x,y):
    return [(x[0]&252)|(y&48)>>4,(x[1]&252)|(y&12)>>2,(x[2]&252)|(y&3)]

class Payload():


    def __init__(self,_img=None,compressionLevel=-1,_content=None):
        self.img = _img
        self.content = _content
        self.compressionLevel = compressionLevel
        self.color = 1
        
        ## DETECT IF COLOR OR BW IMAGE
        if len(np.squeeze(self.img).shape) < 3:
            self.color = 0

        ## DATA VALIDATION: COMPRESSION LEVEL
        if compressionLevel < -1 or compressionLevel > 9:
            raise ValueError("Compression Level must be an integer in [-1,9]")

        ## CHOOSING ACTION TO PERFORM
        if self.img is not None and self.content is None:
            if type(self.img) is not np.ndarray or \
               np.issubdtype(self.img.dtype,np.uint8) is False:
                raise TypeError("Invalid array value type for image. Numpy array expected.")
            self.generate_content()
        elif self.content is not None:
            if type(self.content) is not np.ndarray or\
               np.issubdtype(self.content.dtype,np.uint8) is False:
                raise TypeError("Invalid array value type for content. Numpy array expected.")
            self.reconstruct_img()
        elif self.content is None and self.img is None\
             or self.content is not None and self.img is not None:
            raise ValueError("Both img and content are none. Please call with only one.")
        
    def generate_content(self):
        raw_data = self.unroll()
        #.rfind()
        _t = Timer()
        _t.tic()
        if self.compressionLevel >= 0:
            txt_data = ','.join(list(map(lambda x: str(x),zlib.compress(raw_data,self.compressionLevel))))
            #raw_data = np.frombuffer(zlib.compress(raw_data,self.compressionLevel),np.uint8)
            compressed_str = "True"
        else:
            txt_data = ','.join(list(map(lambda x: str(x),zlib.compress(raw_data,0))))
            #txt_data = ','.join(list(map(lambda x: str(x),raw_data)))
            compressed_str = "False"

        _t.toc()
        print(_t.average_time)
        #txt_data = str(list(map(lambda x: str(x),raw_data)))[1:-1].replace("'","")
        xml_str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><payload type=\"" + self.color_str + "\" size=\"" + str(self.img.shape[0]) + "," + str(self.img.shape[1]) + "\" compressed=\"" + compressed_str + "\">" + txt_data + "</payload>"
        self.content = np.array(list(base64.b64encode(bytes(xml_str,"utf-8"))),dtype=np.uint8)


    def reconstruct_img(self):
        xml_str = base64.b64decode(self.content).decode("utf-8")
        xml_reg = r"<\?xml version=\"1.0\" encoding=\"UTF-8\"\?><payload type=\"(?P<type>(Color|Gray))\" size=\"(?P<len>[0-9]+),(?P<width>[0-9]+)\" compressed=\"(?P<cp>(True|False))\"\>(?P<img>[0-9, ]+)</payload>"
        match = re.match(xml_reg,xml_str)
        if match is None:
            raise ValueError("Invalid string. TODO: Is this how we handle an invalid string?")
        params = match.groupdict()
        if len(params) != 5:
            raise ValueError("Invalid string. TODO: Is this how we handle an invalid string?")

        img_shape = [0,0,0]
        if params["type"] == "Color":
            img_shape[2] = 3
        elif params["type"] == "Gray":
            img_shape[2] = 1
        else:
            raise ValueError("Invalid string. TODO: Is this how we handle an invalid string?")

        img_shape[0] = int(float(params["len"]))
        img_shape[1] = int(float(params["width"]))
        _img = np.array(list(map(lambda x: int(float(x)), params["img"].split(","))),dtype=np.uint8)
        if params["cp"] == "True":
            _img = np.frombuffer(zlib.decompress(_img),dtype=np.uint8)

        if img_shape[2] == 3:
            i = img_shape[0]*img_shape[1]
            _img = np.dstack((_img[:i],_img[1*i:2*i],_img[2*i:]))
            self.img = _img.reshape(img_shape[0],img_shape[1],img_shape[2])
        else:
            self.img = _img.reshape(img_shape[0],img_shape[1])

    def unroll(self):
        if self.color == 1:
            raw_data = np.concatenate((self.img[:,:,0].ravel(),self.img[:,:,1].ravel(),self.img[:,:,2].ravel()))
            self.color_str = "Color"
        else:
            raw_data = self.img.ravel()
            self.color_str = "Gray"
        return raw_data
    

    def b64_encoode(self):
        pass

    def b64_decode(self):
        pass


class Carrier():

    def __init__(self,_img):
        self.img = _img
        self.xml_header = ['01','01','00','00', '01','00','01','00', '00','11','10','01', '00','11','01','00', '01','10','00','10','01','01','01','11','01','11','01','11', '01','10','01','11', '01','10','01','00', '01','10','11','01', '01','01','01','10', '01','11','10','01', '01','10','00','11', '00','11','00','10', '01','10','11','00', '01','11','01','10', '01','10','00','10', '01','10','10','10', '00','11','00','00', '01','10','10','01', '01','00','11','01', '01','01','00','11', '00','11','01','00', '01','11','01','11', '01','00','10','01', '01','10','10','01', '01','00','00','10', '01','10','11','00']
        self.img_shape = self.img.shape
        if len(self.img_shape) < 3:
            self.img_shape.append(1)
        print(self.img_shape,"carrier image shape.")
        if type(self.img) is not np.ndarray or np.issubdtype(self.img.dtype,np.uint8) is False:
            raise TypeError("Incorrect parameter type.")

    def payloadExists(self):
        ## '{0:08b}' --> np.binary_repr(x,width=8)[6:8]
        test_vect = list(map(lambda x,y: np.binary_repr(x,width=8)[6:8] == y,self.img.ravel(),self.xml_header))
        return np.all(test_vec)
    
    def embedPayload(self, payload, override=False):
        if isinstance(payload,Payload) is False or type(payload.content) is not np.ndarray or type(payload.content.dtype) is False:
            raise TypeError("parameter passed contains an incorrect type.")
        if len(payload.content) > np.size(self.img):
            print(len(payload.content))
            print(np.size(self.img))
            raise ValueError("Payload size is larger than what the carrier can hold.")
        if self.payloadExists() is True and override is False:
            raise Exception("Payload exists and override is False.")
        _t = Timer()
        _t.tic()
        this = a_b.reshape((a_b.shape[0]*a_b.shape[1],a_b.shape[2]))[:len(b)]&252|(np.tile(b[:,np.newaxis],3)&[48,12,3])//[8,4,1]
        _t.toc()
        print(_t.average_time)
        _t = Timer()
        _t.tic()
        that = np.concatenate((this,\
            self.img.reshape(\
            (self.img_shape[0]*self.img_shape[1],self.img_shape[2]))[len(b):]))\
            .reshape(self.img_shape[0],self.img_shape[1],self.img_shape[2])
        _t.toc()
        print(_t.average_time)
        return that
        
        # list(map(lambda x,y: \
        #          [(x[0]&252)|(y&48)>>4,(x[1]&252)|(y&12)>>2,(x[2]&252)|(y&3)],\
        #          self.img.reshape((self.img_shape[0]*self.img_shape[1],self.img_shape[2])),\
        #          payload.content)
        ## takes about 0.01 seconds
        # _t.toc()
        # print(_t.average_time)
        # #miter = 0
        # #print(conv_content)
        # # for x,y in zip(self.img.ravel(),payload.content):
        # #     if miter < 10:
        # #         print('{0:08b}'.format(x),'{0:08b}'.format(y))
        # #         print(int('{0:08b}'.format(x)[0:6] + '{0:08b}'.format(y)[6:8],2))
        # #         miter += 1
        # _t = Timer()
        # _t.tic()
        # this = np.concatenate((np.array(list(map(lambda x,y: int('{0:08b}'.format(x)[0:6] + y,2),self.img.ravel(),conv_list)),dtype=np.uint8),self.img.ravel()[len(conv_list):])).reshape(self.img.shape)
        # _t.toc()
        # print(_t.average_time,"this")
        # return this

    def extractPayload(self):
        _t = Timer()
        _t.tic()
        if self.payloadExists() is False:
            raise Exception("No payload present.")
        end_elems = ''.join(list(map(lambda x: '{0:08b}'.format(x)[6:8],self.img.ravel())))
        conv_list = list(map(lambda x: int(x,2),re.findall('....',end_elems)))
        #end_elems = bytearray(end_elems,"utf-8")
        _t.toc()
        #print(end_elems)
        print(_t.average_time,"to bin")

        return end_elems

    def clean(self):
        rand_vals = np.random.randint(0,255,np.size(self.img),dtype=np.uint8)
        return np.array(list(map(lambda x,y: int('{0:08b}'.format(x)[0:6] + '{0:08b}'.format(y)[6:8],2),self.img.ravel(),rand_vals)),dtype=np.uint8).reshape(self.img.shape)




if __name__ == "__main__":

    _t = Timer()
    #filename = "./color.png"
    #filename = "./bw.gif"
    filename = "./nature.jpg"
    myimg = Image.open(filename)
    #myimg.show()

    p_img = np.asarray(myimg, dtype=np.uint8)
    # aim = Image.fromarray(img,"1")
    # immat = aim.load()
    # aim.show()

    a = Payload(p_img,9)

    #emb_array = np.array(list(map(lambda x,y: int('{0:08b}'.format(x)[0:6] + '{0:08b}'.format(y)[6:8],2),a.img,a.content)))
    #print(emb_str.shape)

    filename = "./larger_test.jpg"
    myimg = Image.open(filename)
    c_img = np.asarray(myimg, dtype=np.uint8)

    b = Carrier(c_img)
    _t.tic()
    eimg = b.embedPayload(a)
    _t.toc()    

    print(_t.average_time)    


    b = Carrier(c_img)
    _t.tic()
    b.clean()
    _t.toc()    

    print(_t.average_time)    

    c = Carrier(eimg)
    c.payloadExists()

    c.extractPayload()

    if (False not in (b.img == img)):
        print("SUCCESS")

    #bim = Image.fromarray(b.img*255)
    #bim.show()
    # print("hi")
    
