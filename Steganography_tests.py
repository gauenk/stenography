import time
from os.path import join
import unittest
import numpy as np
from numpy import ndarray as nmat
from scipy.misc import *
from Steganography import *
from checkClean import assertCleaningIsRandom

class ImageAssertion:
    """
    Provides a convenience method for comparing two numpy arrays.
    """
    @staticmethod
    def assertArrayEqual(array1, array2):

        if not isinstance(array1, nmat) or not isinstance(array2, nmat):
            raise AssertionError("One or more of the input parameters are not valid arrays.")

        if array1.shape != array2.shape:
            raise AssertionError("Array shapes do not match.")

        if array1.dtype != array2.dtype:
            raise AssertionError("Array types do not match.")

        if not np.array_equal(array1, array2):
            raise AssertionError("Arrays do not match.")


class SteganographyTestSuite(unittest.TestCase, ImageAssertion):

    folder = "data"

    @staticmethod
    def getXML(path):

        with open(path, "r") as xFile:
            content = xFile.read()

        return content

    def test_PayloadBadInitializer(self):

        with self.subTest(key="Bad Image"):
            img = [[1, 1], [0, 0]]
            self.assertRaises(TypeError, Payload, img)

        with self.subTest(key="Bad Content"):
            self.assertRaises(TypeError, Payload, content=[63, 0, 11, 5])

        with self.subTest(key="Missing Parameters"):
            self.assertRaises(ValueError, Payload)

        with self.subTest(key="Bad Compression Level"):
            img = imread(join(self.folder, "payload1.png"))
            self.assertRaises(ValueError, Payload, img=img, compressionLevel=10)

    def test_PayloadWithImageInput(self):

        referenceArrays = np.load(join(self.folder, "arrays.npz"))
        content1_9 = referenceArrays["content1_9"]
        content1_6 = referenceArrays["content1_6"]
        content2 = referenceArrays["content2"]

        with self.subTest(key="Color Image 1, Level 9"):
            img = imread(join(self.folder, "payload1.png"))
            expectedValue = content1_9
            actualValue = Payload(img, 9).content

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color Image 1, Level 6"):
            img = imread(join(self.folder, "payload1.png"))
            expectedValue = content1_6
            actualValue = Payload(img, 6).content

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color Image 2, Uncompressed"):
            img = imread(join(self.folder, "payload2.png"))
            expectedValue = content2
            actualValue = Payload(img).content

            self.assertArrayEqual(expectedValue, actualValue)

    def test_PayloadWithContentInput(self):

        referenceArrays = np.load(join(self.folder, "arrays.npz"))
        content1_9 = referenceArrays["content1_9"]
        content1_6 = referenceArrays["content1_6"]
        content2 = referenceArrays["content2"]

        with self.subTest(key="Color Image 1, Level 9"):
            img = imread(join(self.folder, "payload1.png"))
            expectedValue = img
            actualValue = Payload(content=content1_9).img

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color Image 1, Level 6"):
            img = imread(join(self.folder, "payload1.png"))
            expectedValue = img
            actualValue = Payload(content=content1_6).img

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color Image 2, Uncompressed"):
            img = imread(join(self.folder, "payload2.png"))
            expectedValue = img
            actualValue = Payload(content=content2).img

            self.assertArrayEqual(expectedValue, actualValue)

    def test_CarrierInitializerAndValidation(self):

        with self.subTest(key="Initializer Check"):
            img = [[1, 1], [0, 0]]
            self.assertRaises(TypeError, Carrier, img)

        with self.subTest(key="Invalid Extraction"):
            c = Carrier(imread(join(self.folder, "carrier2.png")))

            self.assertRaises(Exception, c.extractPayload)

    def test_CarrierImmutability(self):

        with self.subTest(key="After Cleaning"):
            img = imread(join(self.folder, "carrier2.png"))
            ref = img.copy()
            c = Carrier(img)
            c.clean()

            self.assertArrayEqual(ref, c.img)

        with self.subTest(key="After Embedding"):
            img = imread(join(self.folder, "carrier2.png"))
            ref = img.copy()
            c = Carrier(img)
            p = Payload(imread(join(self.folder, "payload2.png")))
            c.embedPayload(p)

            self.assertArrayEqual(ref, c.img)

        with self.subTest(key="After Extraction"):
            img = imread(join(self.folder, "result1_6.png"))
            ref = img.copy()
            c = Carrier(img)
            c.extractPayload()

            self.assertArrayEqual(ref, c.img)

    def test_CarrierCheckingForPayload(self):

        with self.subTest(key="No Payload"):
            img = imread(join(self.folder, "carrier2.png"))
            c = Carrier(img)

            begin = time.clock()
            actualValue = c.payloadExists()
            end = time.clock()

            duration = end - begin

            self.assertTrue(duration < 5 and not actualValue)

        with self.subTest(key="Payload Present"):
            img = imread(join(self.folder, "result1_9.png"))
            c = Carrier(img)
            begin = time.clock()
            actualValue = c.payloadExists()
            end = time.clock()

            duration = end - begin

            self.assertTrue(duration < 5 and actualValue)

    def test_CarrierCleaning(self):

        img = imread(join(self.folder, "carrier1.png"))
        carrier = Carrier(img)

        clean1 = carrier.clean()
        clean2 = carrier.clean()

        with self.subTest(key="Cleaning Once"):

            assertCleaningIsRandom(img, clean1)

        with self.subTest(key="Cleaning Twice"):

            assertCleaningIsRandom(img, clean2)

        with self.subTest(key="Random Clean"):

            self.assertFalse(np.array_equal(clean1, clean2))


    def test_CarrierEmbeddingValidation(self):

        with self.subTest(key="Incorrect Parameter"):
            img = imread(join(self.folder, "payload1.png"))
            c = Carrier(imread(join(self.folder, "carrier1.png")))

            self.assertRaises(TypeError, c.embedPayload, payload=img)

        with self.subTest(key="Payload Exists"):
            p = Payload(imread(join(self.folder, "dummy.png")))
            c = Carrier(imread(join(self.folder, "result1_9.png")))

            self.assertRaises(Exception, c.embedPayload, payload=p)

        with self.subTest(key="Large Payload"):
            p = Payload(imread(join(self.folder, "dummy.png")))
            c = Carrier(imread(join(self.folder, "dummy.png")))

            self.assertRaises(ValueError, c.embedPayload, payload=p)

    def test_CarrierEmbedding(self):

        with self.subTest(key="Compression 9"):
            p = Payload(imread(join(self.folder, "payload1.png")), 9)
            c = Carrier(imread(join(self.folder, "carrier1.png")))

            expectedValue = imread(join(self.folder, "result1_9.png"))
            actualValue = c.embedPayload(p)

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Compression 6"):
            p = Payload(imread(join(self.folder, "payload1.png")), 6)
            c = Carrier(imread(join(self.folder, "carrier1.png")))

            expectedValue = imread(join(self.folder, "result1_6.png"))
            actualValue = c.embedPayload(p)

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="No Compression"):
            p = Payload(imread(join(self.folder, "payload2.png")))
            c = Carrier(imread(join(self.folder, "carrier1.png")))

            expectedValue = imread(join(self.folder, "result2_-1.png"))
            actualValue = c.embedPayload(p)

            self.assertArrayEqual(expectedValue, actualValue)

    def test_CarrierExtraction(self):

        with self.subTest(key="Compression 9"):
            c = Carrier(imread(join(self.folder, "result1_9.png")))

            expectedValue = imread(join(self.folder, "payload1.png"))
            actualValue = c.extractPayload().img

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Compression 6"):
            c = Carrier(imread(join(self.folder, "result1_6.png")))

            expectedValue = imread(join(self.folder, "payload1.png"))
            actualValue = c.extractPayload().img

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="No Compression"):
            c = Carrier(imread(join(self.folder, "result2_-1.png")))

            expectedValue = imread(join(self.folder, "payload2.png"))
            actualValue = c.extractPayload().img

            self.assertArrayEqual(expectedValue, actualValue)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
