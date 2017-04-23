from functools import partial

class Foo():

    def __init__(self):
        self.a = 0

    def myfunc(self):
        print("hi")

def otherFunc(self,ofunc):
    ofunc()
    print("bye")

if __name__=="__main__":

    a = Foo()
    a.myfunc()
    funcType = type(a.myfunc)
    print(funcType)
    pfunc = partial(otherFunc,ofunc=a.myfunc)
    Foo.myfunc = funcType(pfunc,Foo)
    a.myfunc()
    b = Foo()
    b.myfunc()
