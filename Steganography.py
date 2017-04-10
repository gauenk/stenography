#!/usr/bin/env python3.4
import base64,zlib,re,binascii
from scipy.misc import imread,imshow
from PIL import Image
import numpy as np
from timer import Timer ## TODO REMOVE THIS

class Payload():


    def __init__(self,img=None,compressionLevel=-1,content=None):
        self.img = img
        self.content = content
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
        if self.compressionLevel >= 0:
            txt_data = np.frombuffer(zlib.compress(raw_data,self.compressionLevel),dtype=np.uint8)
            compressed_str = [ 84, 114, 117, 101] # True in numbers
        else:
            txt_data = raw_data
            compressed_str = [ 70,  97, 108, 115, 101] # False in numbers

        #print(np.size(self.img),len(raw_data),len(txt_data))
        xml_str = self.make_xml(txt_data,compressed_str)
        xml_up = np.unpackbits(xml_str)
        padding = np.zeros(6-len(xml_up)%6,dtype=np.uint8)
        if len(padding) == 6:
            padding = np.zeros(0,dtype=np.uint8)
        this = np.right_shift(np.packbits(\
                 np.append(xml_up,padding).reshape((len(xml_up)+len(padding))//6,6)\
                 ,axis=1),2).ravel()
        self.content = this

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
            self.color_str = [ 67, 111, 108, 111, 114] # Color
        else:
            raw_data = self.img.ravel()
            self.color_str = [ 71, 114,  97, 121] # Gray
        return raw_data
    
    def make_xml(self,txt_data,compressed_str):
        return np.concatenate(([ 60,  63, 120, 109, 108,  32,\
                                 118, 101, 114, 115, 105, 111, \
                                 110, 61,  34,  49,  46,  48,  \
                                 34,  32, 101, 110,  99, 111, 100,\
                                 105, 110, 103,  61,  34,  85,  84,\
                                 70,  45,  56,  34,  63,  62,  60,\
                                 112,  97, 121, 108, 111,  97, 100,\
                                 32, 116, 121, 112, 101,  61,  34]\
                  ,self.color_str\
                  ,[ 34,  32, 115, 105, 122, 101,  61,  34]\
                  ,[self.img.shape[0]]\
                  ,[44],[self.img.shape[1]]\
                  , [ 34,  32,  99, 111, 109, 112, 114, 101, 115, 115, 101, 100,  61,  34]\
                  , compressed_str\
                  , [34,62]\
                  , txt_data\
                  , [ 60,  47, 112,  97, 121, 108, 111,  97, 100,  62]),axis=0).astype(np.uint8) # end of xml_str
    

class Carrier():

    def __init__(self,_img):
        self.img = _img
        self.xml_header = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
       0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0,
       0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
       1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
       0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
       0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
       1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1,
       0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
       1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
       1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
       1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        #self.xml_tail = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
        # 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
        # 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
        # 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],dtype=np.uint8)
        self.color=True
        if len(np.squeeze(self.img.shape)) < 3:
            self.color=False
        self.img_shape = self.img.shape
        if len(self.img_shape) < 3:
            self.img_shape.append(1)
        #print(self.img_shape,"carrier image shape.")
        if type(self.img) is not np.ndarray or np.issubdtype(self.img.dtype,np.uint8) is False:
            raise TypeError("Incorrect parameter type.")

    def payloadExists(self):
        up_img = np.unpackbits(self.img.ravel()[:208])
        if np.array_equal(up_img[6:len(self.xml_header)*4:8],self.xml_header[0::2]) and \
           np.array_equal(up_img[7:len(self.xml_header)*4:8],self.xml_header[1::2]):
            return True
        return False
        ## '{0:08b}' --> np.binary_repr(x,width=8)[6:8]
        # count = 0
        # for x,y in zip(self.img.ravel(),self.xml_header):
        #     if count < 5:
        #         count +=1
        #         print(x,np.binary_repr(x,width=8),y)

        # test_vect = list(map(lambda x,y: np.binary_repr(x,width=8)[6:8] == y,self.img.ravel(),self.xml_header))
        # #test_vect = list(map(lambda x,y: np.binary_repr(x,width=8)[6:8] == y,self.img.ravel(),self.xml_header))
        # print(test_vect)
        # return np.all(test_vect)
    
    def embedPayload(self, payload, override=False):
        content_len = len(payload.content)
        img_size = np.size(self.img)
        if self.color is True:
            img_size = img_size // 3
        if isinstance(payload,Payload) is False or type(payload.content) is not np.ndarray or type(payload.content.dtype) is False:
            raise TypeError("parameter passed contains an incorrect type.")
        if content_len > img_size:
            # print(len(payload.content))
            # print(np.size(self.img))
            raise ValueError("Payload size is larger than what the carrier can hold.")
        if self.payloadExists() is True and override is False:
            raise Exception("Payload exists and override is False.")
        #print(content_len)
        #print(np.size(self.img))
        this = self.img.reshape((self.img_shape[0]*self.img_shape[1],self.img_shape[2]))[:content_len]&252|(np.tile(payload.content[:,np.newaxis],3)&[48,12,3])//[16,4,1]
        # up_img = np.unpackbits(np.copy(self.img))
        # up_cont = np.unpackbits(payload.content)
        # up_cont = up_cont.reshape(len(up_cont)//8,8)[:,2:].ravel()
        # up_img[6:len(up_cont)//2*8:8] = up_cont[::2]
        # up_img[7:len(up_cont)//2*8:8] = up_cont[1::2]
        # up_img = np.packbits(up_img).reshape(self.img_shape[0],self.img_shape[1],self.img_shape[2])
        # up_img = np.squeeze(up_img)
        that = np.concatenate((this,\
            self.img.reshape(\
            (self.img_shape[0]*self.img_shape[1],self.img_shape[2]))[content_len:]))\
            .reshape(self.img_shape[0],self.img_shape[1],self.img_shape[2])\
            .astype(np.uint8)
        #that = up_img
        # print((np.tile(payload.content[:,np.newaxis],3)&[48,12,3])//[16,4,1],"content")
        # count = 0
        # print(payload.content)
        # for i in ((np.tile(payload.content[:,np.newaxis],3)&[48,12,3])//[16,4,1]):
        #     if count < 5:
        #         count +=1 
        #         print(bin(i[0]),bin(i[1]),bin(i[2]))
        # print(np.unpackbits(self.img[0][0]).reshape(3,8)[:,6:])
        # print(np.unpackbits(that[0][0]).reshape(3,8)[:,6:])
        # print(np.unpackbits(payload.content[0:2]).reshape(2,8)[:,2:])
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
        if self.payloadExists() is False:
            raise Exception("No payload present.")
        
        up_img = np.unpackbits(self.img)
        _mid = np.ravel(np.column_stack((up_img[6::8],up_img[7::8])).astype(np.uint8))
        if len(_mid) % 6 != 0:
            print("PROBLEM",len(_mid) % 6)
        #print(np.packbits(_mid.reshape((len(_mid)//6,6)),axis=1))
        #print(_mid,"mid")
        _out = np.right_shift(np.packbits(_mid.reshape((len(_mid)//6,6)),axis=1),2).ravel()
        #print(_out)
        #_out = np.packbits(_mid,axis=0)
        return _out
        # up_cont = up_cont.reshape(len(up_cont)//8,8)[:,2:].ravel()

        # up_img[7:len(up_cont)//2*8:8] = up_cont[1::2]

        #end_elems = ''.join(list(map(lambda x: '{0:08b}'.format(x)[6:8],self.img.ravel())))
        #conv_list = list(map(lambda x: int(x,2),re.findall('....',end_elems)))
        #end_elems = bytearray(end_elems,"utf-8")

        #return end_elems

    def clean(self):
        rand_vals = np.random.randint(0,255,np.size(self.img),dtype=np.uint8)
        return np.array(list(map(lambda x,y: int('{0:08b}'.format(x)[0:6] + '{0:08b}'.format(y)[6:8],2),self.img.ravel(),rand_vals)),dtype=np.uint8).reshape(self.img.shape)




def print_time_dict(_t):
    for i,v in _t.items():
        print(i,v.average_time)

def test_payload_prep():
    #filename = "./nature.jpg"
    filename = "./color.png"
    myimg = Image.open(filename)
    p_img = np.asarray(myimg, dtype=np.uint8)
    _t = {'total': Timer(),'to_content': Timer(),'to_img': Timer()}
    _t['total'].tic()
    _t['to_content'].tic()
    a = Payload(p_img,9)
    _t['to_content'].toc()
    _t['to_img'].tic()
    a = Payload(content=a.content)
    _t['to_img'].toc()
    _t['total'].toc()
    print_time_dict(_t)
    print(np.array_equal(p_img,a.img))

def test_carrier_clean():
    filename = "./nature.jpg"
    myimg = Image.open(filename)
    c_img = np.asarray(myimg, dtype=np.uint8)
    _t = Timer()
    b = Carrier(c_img)
    _t.tic()
    b.clean()
    _t.toc()    
    print("clean time",_t.average_time)

def embed_payload():
    #filename = "./nature.jpg"

    payload_filename = "./nature.jpg"
    #payload_filename = "./color.png"
    carrier_filename = "./larger_test.jpg"

    payload_img = imread(payload_filename)
    carrier_img = imread(carrier_filename)

    _t = {'payload': Timer(),'embed': Timer(),\
          'payload_exists_i':Timer(),'payload_exists_e':Timer(),\
          'extract_payload':Timer()}

    _t['payload'].tic()
    a = Payload(payload_img,9) ## create content to embed
    _t['payload'].toc()

    b = Carrier(carrier_img)    ## create Carrier instance

    _t['payload_exists_i'].tic()
    print(b.payloadExists() == False)
    _t['payload_exists_i'].toc()


    _t['embed'].tic()
    eimg = b.embedPayload(a) ## embed a into b
    _t['embed'].toc()

    #imshow(carrier_img) ## show both images to see the difference
    #imshow(eimg) ## show to see the difference

    b_e = Carrier(eimg) # create carrier WITH embedded image

    _t['payload_exists_e'].tic()
    print(b_e.payloadExists() == True)
    _t['payload_exists_e'].toc()


    _t['extract_payload'].tic()
    ex_img = b_e.extractPayload()
    _t['extract_payload'].toc()

    idx = range(1790,1804)
    print(np.array_equal(ex_img[:len(a.content)],a.content),"extract") ## extraction should be equal to embedded content
    
    print_time_dict(_t)

if __name__ == "__main__":


    #test_payload_prep()
    #test_carrier_clean()
    embed_payload()

