#!/usr/bin/env python3.4
import zlib,re,binascii
from scipy.stats import bernoulli
from scipy.misc import imread,imshow
from PIL import Image
import numpy as np


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

        k = txt_data.astype("<U3").view(np.dtype("a4"))
        til = np.tile(",",len(k)//3)[:,np.newaxis]
        l = np.hstack((k.reshape(len(k)//3,3),til)).ravel()[:-1]
        m = np.delete(l,np.where(l=='')).view(int)[::2].astype(np.uint8)
        xml_str = self.make_xml(m,compressed_str)
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
            raise ValueError("Invalid string. Regex did not match anything.")
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
        self.xml_header_bits = [1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1]
        self.xml_header_values = [15,3,61,56,27,22,48,32,29,38,21,50,28]
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
        if np.array_equal(up_img[7:len(self.xml_header_bits)*4:8],self.xml_header_bits[0::2]) and \
           np.array_equal(up_img[6:len(self.xml_header_bits)*4:8],self.xml_header_bits[1::2]):
            return True
        return False

    def embedPayload(self, payload, override=False):
        img_size = np.size(self.img)//3 ## TODO: test if true for BW images
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
        up_img = np.copy(self.img).ravel()
        lsb = (up_img&3).reshape(-1,3)
        array = np.left_shift(lsb[:,2],4) | np.left_shift(lsb[:,1],2) | lsb[:,0]
        return Payload(content=array)

    def clean(self):
        up_img = np.copy(self.img).ravel()
        rand_vals = np.random.randint(0,2,len(up_img)).astype(np.uint8)
        up_img = up_img&252|rand_vals
        return up_img.reshape(self.img.shape)

    def extractPayloadAdvanced(self):
        # 99 is smallest valid xml structure w/ image
        max_i = np.size(self.img)-99
        max_n = np.ceil(np.size(self.img)//99)
        up_img = np.copy(self.img).ravel()
        lsb = (up_img&3).reshape(-1,3)
        array = np.left_shift(lsb[:,2],4) | np.left_shift(lsb[:,1],2) | lsb[:,0]
        
        
        f = np.where(array==self.xml_header_values[0])[0]
        s = np.where(array[f[0]:]==self.xml_header_values[1])[0]+f[0]
        t = np.where(array[s[0]:]==self.xml_header_values[2])[0]+s[0]
        fr = np.where(array[t[0]:]==self.xml_header_values[4])[0]+t[0]
        min_len = np.min([len(f),len(s),len(t),len(fr)])
        i = 0
        n = 1
        s_mat = np.tile(s[:,np.newaxis],len(f))
        ds_mat = s_mat - f
        t_mat = np.tile(t[:,np.newaxis],len(s))
        dt_mat = t_mat - s
        loc = np.where(np.equal(dt_mat,ds_mat))
        print(loc)
        

        # print(len(f),len(s))

        # ab = np.where(np.equal((s[:min_len]-f[:min_len]),(fr[:min_len]-t[:min_len]),(t[:min_len]-s[:min_len])))
        # print(f[33])
        # print(s[33])
        # print(t[33])
        # print(ab,"av")

        # if (s[0] - f[0]) != (t[0] - s[0]):
        #     print("HERE")
        #     print(f[0],s[0],t[0],fr[0])
        #     s = np.where(array[s[0]+1:]==self.xml_header_values[1])[0]+s[0]+1
        #     t = np.where(array[s[0]:]==self.xml_header_values[2])[0]+s[0]
        #     fr = np.where(array[t[0]:]==self.xml_header_values[4])[0]+t[0]
        # print(f[0],s[0],t[0],fr[0])
        
        f_array = array


        return Payload(content=f_array)


    def embedPayloadAdvanced(self, payload, initialPoint, step):
        if isinstance(payload,Payload) is False or type(payload.content) is not np.ndarray or type(payload.content.dtype) is False or type(initialPoint) is not tuple or type(step) is not int:
            raise TypeError("parameter passed contains an incorrect type.")
        img_size =(self.img_shape[0] - initialPoint[0])\
                    * (self.img_shape[1] - initialPoint[1])
        if step != 0:
            img_size = int(np.ceil(img_size/step))
        if self.color is True:
            img_size = 3*img_size
        content_len = len(payload.content)
        if content_len > img_size:
            raise ValueError("Payload size is larger than what the carrier can hold.")

        if self.color is True:
            bimg = np.copy(self.img)
            start_idx = initialPoint[0]*bimg.shape[1]+initialPoint[1]
            rbimg = bimg.reshape((-1,3))[start_idx:start_idx+content_len*step:step]
            rbimg = rbimg&252|(np.tile(payload.content[:,np.newaxis],3)&[3,12,48])//[1,4,16]
            this = bimg.reshape(-1,3)
            this[start_idx:start_idx+content_len*step:step,:] = rbimg
            that = this.reshape(bimg.shape).astype(np.uint8)
        else:
            bimg = np.copy(self.img).ravel()[0:np.size(self.img)-(np.size(self.img)%3)] 
            assert(len(bimg)%3 == 0)
            start_idx = initialPoint[0]*bimg.shape[1]+initialPoint[1]
            rbimg = bimg.reshape((-1,3))[start_idx:start_idx+content_len*step:step]
            rbimg = rbimg&252|(np.tile(payload.content[:,np.newaxis],3)&[3,12,48])//[1,4,16]
            this = bimg.reshape(-1,3)
            this[start_idx:start_idx+content_len*step:step,:] = rbimg
            that = this.reshape(bimg.shape).astype(np.uint8)
        return that


if __name__ == "__main__":
    print("Main function")

