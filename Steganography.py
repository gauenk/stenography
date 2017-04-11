#!/usr/bin/env python3.4
import base64,zlib,re,binascii
from scipy.stats import bernoulli
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
        ## DATA VALIDATION: COMPRESSION LEVEL
        if compressionLevel < -1 or compressionLevel > 9:
            raise ValueError("Compression Level must be an integer in [-1,9]")
        ## CHOOSING ACTION TO PERFORM
        if self.img is not None and self.content is None:
            if type(self.img) is not np.ndarray or \
               np.issubdtype(self.img.dtype,np.uint8) is False:
                ## DETECT IF COLOR OR BW IMAGE
                raise TypeError("Invalid array value type for image. Numpy array expected.")
            if len(np.squeeze(self.img).shape) < 3:
                self.color = 0
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
        # _t = Timer()
        # _t.tic()
        #txt_data = np.fromstring(','.join(map(str, txt_data)),np.uint8)
        k = txt_data.astype("<U3").view(np.dtype("a4"))
        # _t.toc()
        # print(_t.average_time,"viewing")
        # _t = Timer()
        # _t.tic()
        til = np.tile(",",len(k)//3)[:,np.newaxis]
        # _t.toc()
        # print(_t.average_time,"tile")
        # _t = Timer()
        # _t.tic()
        l = np.hstack((k.reshape(len(k)//3,3),til)).ravel()[:-1]
        # _t.toc()
        # print(_t.average_time,"reshape")
        # _t = Timer()
        # _t.tic()
        m = np.delete(l,np.where(l=='')).view(int)[::2].astype(np.uint8)
        # _t.toc()
        # print(_t.average_time)

        xml_str = self.make_xml(m,compressed_str)
        #print(len(xml_str),"len(xml_str)")
        xml_up = np.unpackbits(xml_str)
        padding = np.zeros(6-len(xml_up)%6,dtype=np.uint8)
        if len(padding) == 6:
            padding = np.zeros(0,dtype=np.uint8)
        this = np.right_shift(np.packbits(\
                 np.append(xml_up,padding).reshape((len(xml_up)+len(padding))//6,6)\
                 ,axis=1),2).ravel()
        self.content = this


    def reconstruct_img(self):
        up_con = np.unpackbits(self.content)
        padding = np.zeros(8-len(up_con)%8,dtype=np.uint8)
        if len(padding) == 8:
            padding = np.zeros(0,dtype=np.uint8)
        up_con = np.concatenate((up_con,padding))
        up_con = up_con.reshape(-1,8)[:,2:].ravel()
        padding = np.zeros(8-len(up_con)%8,dtype=np.uint8)
        if len(padding) == 8:
            padding = np.zeros(0,dtype=np.uint8)
        up_con = np.concatenate((up_con,padding))
        up_con = up_con.reshape(-1,8)
        up_con = np.packbits(up_con,axis=1).ravel()
        xml_str = ''.join(list(map(lambda x: chr(x),up_con[0:100]))) # max image size supported by Adobe is 300000 by 300000... we assume that is max
        xml_reg = r"<\?xml version=\"1.0\" encoding=\"UTF-8\"\?><payload type=\"(?P<type>(Color|Gray))\" size=\"(?P<len>[0-9]+),(?P<width>[0-9]+)\" compressed=\"(?P<cp>(True|False))\">"
        
        match = re.match(xml_reg,xml_str)
        if match is None:
            raise ValueError("Invalidd string. Regex did not match anything.")
        params = match.groupdict()
        if len(params) != 4:
            raise ValueError("Invalid string. Unexpected number of matches.")
        img_shape = [0,0,0]
        if params["type"] == "Color":
            img_shape[2] = 3
        elif params["type"] == "Gray":
            img_shape[2] = 1
        else:
            raise ValueError("Invalid string. TODO: Is this how we handle an invalid string?")
        img_shape[0] = int(float(params["len"]))
        img_shape[1] = int(float(params["width"]))
        start_idx = np.where(up_con[:100]==62)[0][-1]+1
        end_idx = np.where(up_con[100:]==60)[0][0]+100
        this = "[" + ''.join(map(lambda x: chr(x),up_con[start_idx:end_idx])) + "]"
        _img = np.array(eval(this),dtype=np.uint8)
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
        #print(len(txt_data),"len(txt_data)")
        #print(txt_data[:10],txt_data[-10:])
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
                  ,list(map(lambda x: ord(x),list(str(self.img.shape[0]))))\
                  ,[44],list(map(lambda x: ord(x),list(str(self.img.shape[1]))))\
                  , [ 34,  32,  99, 111, 109, 112, 114, 101, 115, 115, 101, 100,  61,  34]\
                  , compressed_str\
                  , [34,62]\
                  , txt_data\
                  , [ 60,  47, 112,  97, 121, 108, 111,  97, 100,  62]),axis=0).astype(np.uint8) # end of xml_str
    

class Carrier():

    def __init__(self,_img):
        self.img = _img
        #self.xml_header = [1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,1,0,1,1,0,0,1,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,1]
        self.xml_header = [1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1]
        self.color=True
        if type(self.img) is not np.ndarray or np.issubdtype(self.img.dtype,np.uint8) is False:
            raise TypeError("Incorrect parameter type.")
        if len(np.squeeze(self.img.shape)) < 3:
            self.color=False
        self.img_shape = self.img.shape
        if len(self.img_shape) < 3:
            self.img_shape += (1,)

    def payloadExists(self):
        up_img = np.unpackbits(self.img.ravel()[:208])
        if np.array_equal(up_img[7:len(self.xml_header)*4:8],self.xml_header[0::2]) and \
           np.array_equal(up_img[6:len(self.xml_header)*4:8],self.xml_header[1::2]):
            return True
        return False

    def embedPayload(self, payload, override=False):
        img_size = np.size(self.img)
        if self.color is True:
            img_size = img_size // 3
        if isinstance(payload,Payload) is False or type(payload.content) is not np.ndarray or type(payload.content.dtype) is False:
            raise TypeError("parameter passed contains an incorrect type.")
        content_len = len(payload.content)
        if content_len > img_size:
            raise ValueError("Payload size is larger than what the carrier can hold.")
        if self.payloadExists() is True and override is False:
            raise Exception("Payload exists and override is False.")
        if self.color is True:
            this = self.img.reshape((self.img_shape[0]*self.img_shape[1],self.img_shape[2]))[:content_len]&252|(np.tile(payload.content[:,np.newaxis],3)&[3,12,48])//[1,4,16]
            that = np.concatenate((this,\
                self.img.reshape(\
                (self.img_shape[0]*self.img_shape[1],self.img_shape[2]))[content_len:]))\
                .reshape(self.img_shape[0],self.img_shape[1],self.img_shape[2])\
                .astype(np.uint8)
        else:
            b_img = self.img.ravel()[0:np.size(self.img)-(np.size(self.img)%3)] 
            assert(len(b_img)%3 == 0)
            this = b_img.reshape((self.img_shape[0]*self.img_shape[1]//3,3))[:content_len]&252|(np.tile(payload.content[:,np.newaxis],3)&[3,12,48])//[1,4,16]
            that = np.concatenate((this,\
                b_img.reshape(\
                (b_img.shape[0]//3,3))[content_len:]))\
                .reshape(self.img_shape[0],self.img_shape[1])\
                .astype(np.uint8)
        return that

    def extractPayload(self):
        if self.payloadExists() is False:
            raise Exception("No payload present.")
        up_img = self.img.ravel()
        lsb = (up_img&3).reshape(-1,3)
        array = np.left_shift(lsb[:,2],4) | np.left_shift(lsb[:,1],2) | lsb[:,0]
        return Payload(content=array)

    def clean(self):
        up_img = np.copy(self.img).ravel()
        rand_vals = np.random.randint(0,2,len(up_img),dtype=np.uint8)
        up_img = up_img&252|rand_vals
        return up_img.reshape(self.img.shape)


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

    payload_filename = "./medium.png"
    #payload_filename = "./color.png"
    carrier_filename = "./world.jpg"
    #carrier_filename = "./nature.jpg"

    payload_img = imread(payload_filename)
    carrier_img = imread(carrier_filename)

    _t = {'payload': Timer(),'embed': Timer(),\
          'payload_exists_i':Timer(),'payload_exists_e':Timer(),\
          'extract_payload':Timer(),'generate':Timer(),
           'ex_gen':Timer()}

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
    ex_payload = b_e.extractPayload()
    _t['extract_payload'].toc()

    idx = range(1790,1804)
    print(np.array_equal(ex_payload.img[:len(a.content)],a.img),"extract") ## extraction should be equal to embedded content

    _t['generate'].tic()
    c = Payload(content=a.content)
    _t['generate'].toc()

    print(np.array_equal(c.img,a.img),"generate")

    _t['ex_gen'].tic()
    print("EX_GEN\n\n")
    c = Payload(content=ex_payload.content)
    _t['ex_gen'].toc()

    print(np.array_equal(c.img,a.img),"ex_gen")


    print_time_dict(_t)
##BOTTLE NECKS:
# extract payload
# ex_gen


if __name__ == "__main__":


    #test_payload_prep()
    #test_carrier_clean()
    embed_payload()

