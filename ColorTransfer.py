import numpy as np
import cv2
from VirtualEffect import VirtualEffect
from math import sqrt
import copy

__allowed_modes__ = {"GrayScale", "YUV", "BGR", "RGB", "Custom"}

class ColorTransfer(VirtualEffect):
    """
    Unary operator that precomputes on a image to transfer the color to another one.
    According to Color transfer between images E. Reinhard et al. Publisher: IEEE
    Clips the transposition of mean and std from YUV (lab space) representation
    """

    def __init__(self) -> None:
        """
        Init an instance of a Color Tranfer Effect
        :key meta: dict storing information about the effect to be applied
        :key meta.__desc__: stores a description of the Effect
        :key meta.__len_op__: 1 as an Unary operator
        :key meta.__len__: size of the images GrayScale 1, images RGB 3
        :key _mean: mean of the reference image channels in YUV
        :key _std: std of the reference image channels in YUV
        :return: None
        """
        super().__init__()
        self.meta["__desc__"] = "Unary operator that precomputes on a image to transfer the color to another one.\nAccording to Color transfer between images E. Reinhard et al. Publisher: IEEE\nClips the transposition of mean and std from YUV (lab space) representation"
        # Unary operator
        self.meta["__len_op__"] = 1
        # Color Transfer Parameters
        self._mean = []
        self._std = []

    def __repr__(self) -> str :
        raise NotImplementedError

    def apply(self, img : np.ndarray, alpha : float = 1, mode : str = "YUV") -> None:
        """
        Apply the unary Operator of Color transfer to an image
        :rtype: a numpy array with the effect applied on img
        :param img: objects corresponding to the number of parameter of the operator (unary for 1),
        on which the effect will be applied
        :param alpha: percentage of application of the Color transfer Effect
        :param mode: input type of images, supports YUV | BGR | RGB
        :return: None, Transforms map_able_objects
        assert len(map_able_objects) == self.meta[
            "__len_op__"], "Operator must apply on dimension objects of same dimension"
        assert self.meta["__len__"] == len(map_able_object), "Src and Dst images must have the same number of channels"
        """

        assert mode in __allowed_modes__, "Unsupported mode {}, please refer to the manual or implement it yourself\n\tSupported Modes :\n{}".format(
            mode, __allowed_modes__)

        assert isinstance(img, np.ndarray), "Src img must have a numpy.ndarray type."

        if mode == "YUV" or mode == "Custom" :
            # Direct Calculation
            s = img.shape
            n = s[0] * s [1]
            mean = [0 for l in range(s[2])]
            std = [0 for l in range(s[2])]
            # Mean calculation
            for i in range(s[0]):
                for j in range(s[1]):
                    for k in range(s[2]):
                        mean[k] += img[i,j,k]
            mean = [x / n for x in mean]
            # STD calculation :
            tmp_square = 0
            for i in range(s[0]):
                for j in range(s[1]):
                    for k in range(s[2]):
                        tmp_square = img[i,j,k] - mean[k]
                        std[k] += tmp_square * tmp_square
            std = [x / n for x in std]
            # Application of the effect
            for i in range(s[0]):
                for j in range(s[1]):
                    for k in range(s[2]):
                        tmp_apply = (img[i,j,k] - mean[k]) * sqrt(self._std[k] / std[k]) + self._mean[k]
                        # clip
                        value = img[i, j, k] * (1 - alpha) + alpha * tmp_apply
                        if value <= 0:
                            img[i, j, k] = 0
                        elif value >= 255 :
                            img[i, j, k] = 255
                        else :
                            img[i, j, k] = value
            test = copy.deepcopy(img)
            return cv2.cvtColor(test, cv2.COLOR_YUV2BGR)
        elif mode == "BGR" :
            # Conversion from cv2
            return self.apply(cv2.cvtColor(img, cv2.COLOR_BGR2YUV), alpha, "YUV")
        elif mode == "RGB" :
            # Conversion from cv2
            return cv2.cvtColor(self.apply(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YUV), alpha, "YUV"), cv2.COLOR_BGR2RGB)
        elif mode == "GrayScale" :
            # in that case it is a simple luminosity / contrast match
            pass
        return img


    def precompute(self, reference : np.ndarray, mode : str = "YUV") -> None:
        """
        Precomputes the Color Transfer Operator on an iterable object (must be in the YUV space)
        :param reference: image as iterable object as reference for the color transfer (mean and std)
        :param mode: input type of images, supports YUV | BGR | RGB, mind that imread of openCV represents images in BGR
        :return: None, Will store in self all necessary data for the operator top be functional
        """
        assert mode in __allowed_modes__, "Unsupported mode {}, please refer to the manual or implement it yourself\n\tSupported Modes :\n{}".format(mode, __allowed_modes__)

        assert isinstance(img, np.ndarray), "Src img must have a numpy.ndarray type."

        if mode == "YUV" or mode == "Custom" :
            # Direct Calculation
            s = img.shape
            n = s[0] * s[1]
            self._mean = [0 for l in range(s[2])]
            self._std = [0 for l in range(s[2])]
            self.meta["__len__"] = s[2]
            # Mean calculation
            for i in range(s[0]):
                for j in range(s[1]):
                    for k in range(s[2]):
                        self._mean[k] += img[i, j, k]
            self._mean = [x / n for x in self._mean]
            # STD calculation :
            tmp_square = 0
            for i in range(s[0]):
                for j in range(s[1]):
                    for k in range(s[2]):
                        tmp_square = img[i, j, k] - self._mean[k]
                        self._std[k] += tmp_square * tmp_square
            self._std = [x / n for x in self._std]
        elif mode == "BGR" :
            # Conversion from cv2
            self.precompute(cv2.cvtColor(reference, cv2.COLOR_BGR2YUV), "YUV")
        elif mode == "RGB" :
            # Conversion from cv2
            self.precompute(cv2.cvtColor(cv2.cvtColor(reference, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YUV), "YUV")
        elif mode == "GrayScale" :
            # in that case it is a simple luminosity / contrast match
            pass

if __name__ == "__main__" :
    ref = cv2.imread("Sky.jpg")
    img = cv2.imread("photo.png")
    ref_rez = cv2.resize(ref, (int(ref.shape[1] * 480 / ref.shape[0]), 480))
    img_rez = cv2.resize(img, (int(img.shape[1] * 480 / img.shape[0]), 480))
    # fixed height 480 pixels
    FX = ColorTransfer()
    FX.precompute(ref_rez, "BGR")
    img_effect = FX.apply(img_rez, 1, "BGR")
    cv2.imshow("Reference", ref_rez)
    cv2.imshow("Image ref", img_rez)
    cv2.imshow("Image Effect", img_effect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

