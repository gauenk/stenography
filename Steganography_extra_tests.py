import time
from os.path import join
import unittest
import numpy as np
from numpy import ndarray as nmat
from scipy.misc import *
from Steganography import *

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


class SteganographyExtraTestSuite(unittest.TestCase, ImageAssertion):

    folder = "extra_data"

    def test_GrayCarrierEmbedding(self):

        with self.subTest(key="Gray in Gray"):
            p = Payload(imread(join(self.folder, "payload3.png")), 3)
            c = Carrier(imread(join(self.folder, "carrier3.png")))

            expectedValue = imread(join(self.folder, "result3_3.png"))
            actualValue = c.embedPayload(p)

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color in Gray"):
            p = Payload(imread(join(self.folder, "payload4.png")), 7)
            c = Carrier(imread(join(self.folder, "carrier3.png")))

            expectedValue = imread(join(self.folder, "result4_7.png"))
            actualValue = c.embedPayload(p)

            self.assertArrayEqual(expectedValue, actualValue)

    def test_GrayCarrierExtraction(self):

        with self.subTest(key="Gray in Gray"):
            c = Carrier(imread(join(self.folder, "result3_3.png")))

            expectedValue = imread(join(self.folder, "payload3.png"))
            actualValue = c.extractPayload().img

            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key="Color in Gray"):
            c = Carrier(imread(join(self.folder, "result4_7.png")))

            expectedValue = imread(join(self.folder, "payload4.png"))
            actualValue = c.extractPayload().img

            self.assertArrayEqual(expectedValue, actualValue)

    def test_EmbeddingPerformance(self):

        durations = []

        for iteration in range(5):
            begin = time.clock()
            referenceArray = imread(join(self.folder, "result2_9.png"))
            p = Payload(imread(join(self.folder, "payload2.png")), 9)
            c = Carrier(imread(join(self.folder, "carrier2.png")))
            result = c.embedPayload(p)

            output = np.array_equal(referenceArray, result)
            end = time.clock()

            durations.append(end - begin)

        averageDuration = sum(durations) / len(durations)
        durationString = [format(i, "2.3f") for i in durations]

        print("Execution Times = {}".format(durationString))
        print("Average Execution Time = {:2.3f} sec".format(averageDuration))

        self.assertLess(averageDuration, 2.0)

    def test_ExtractionPerformance(self):

        durations = []

        for iteration in range(5):
            begin = time.clock()

            referenceArray = imread(join(self.folder, "payload2.png"))
            c = Carrier(imread(join(self.folder, "result2_9.png")))
            result = c.extractPayload()

            output = np.array_equal(referenceArray, result.img)

            end = time.clock()

            durations.append(end - begin)

        averageDuration = sum(durations) / len(durations)
        durationString = [format(i, "2.3f") for i in durations]

        print("Execution Times = {}".format(durationString))
        print("Average Execution Time = {:2.3f} sec".format(averageDuration))

        self.assertLess(averageDuration, 3.0)

    def test_VariableEmbedding(self):

        p = Payload(imread(join(self.folder, "payload5.png")), 8)
        c = Carrier(imread(join(self.folder, "carrier2.png")))

        expectedValue = imread(join(self.folder, "result5.png"))
        actualValue = c.embedPayloadAdvanced(p, (1, 1000), 10)

        self.assertArrayEqual(expectedValue, actualValue)


    def test_VariableExtraction(self):
        with self.subTest(key = "Known Sample"):
            c = Carrier(imread(join(self.folder, "result5.png")))

            expectedValue = imread(join(self.folder, "payload5.png"))
            begin = time.clock()
            actualValue = c.extractPayloadAdvanced().img
            end = time.clock()

            duration = end - begin
            print("Variable Extraction 1 Duration = {0:2.4f} sec".format(duration))
            self.assertArrayEqual(expectedValue, actualValue)

        with self.subTest(key = "Unknown Sample"):
            c = Carrier(imread(join(self.folder, "result6.png")))

            expectedValue = imread(join(self.folder, "payload6.png"))
            begin = time.clock()
            actualValue = c.extractPayloadAdvanced().img
            end = time.clock()

            duration = end - begin
            print("Variable Extraction 2 Duration = {0:2.4f} sec".format(duration))

            self.assertArrayEqual(expectedValue, actualValue)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
