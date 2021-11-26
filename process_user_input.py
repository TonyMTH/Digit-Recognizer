import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


class Process:
    def __init__(self, img, original_image, kernel_size, min_thresh, n, dim):
        self.img = img
        self.kernel_size = kernel_size
        self.min_thresh = min_thresh
        self.n = n
        self.dim = dim
        self.original_image = original_image

    def read_to_black_white_img(self):
        im = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        return im

    def convert_to_black_background(self):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        (thresh, img) = cv2.threshold(self.read_to_black_white_img(), self.min_thresh, 255, cv2.THRESH_BINARY_INV)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def gety1y2(self):
        img = self.convert_to_black_background()
        r = img.shape[0]
        y1, y2 = None, None
        for i in range(r):
            if any(img[i, :] != 0) and (y1 is None):
                y1 = i
            if all(img[i, :] == 0) and (y1 is not None):
                y2 = i
                break
        return y1 if y1 is not None else 0, y2 if y2 is not None else r - 1

    def getx1x2(self):
        img = self.convert_to_black_background()
        c = img.shape[1]
        x1, x2 = None, None
        for i in range(c):
            if any(img[:, i] != 0) and (x1 is None):
                x1 = i
            if all(img[:, i] == 0) and (x1 is not None):
                x2 = i
                break
        return x1 if x1 is not None else 0, x2 if x2 is not None else c - 1

    def crop(self):
        img = self.convert_to_black_background()

        x1, x2 = self.getx1x2()
        y1, y2 = self.gety1y2()
        # img = img[y1 - 1:y2 + 1, x1 - 1:x2 + 1]
        img = np.pad(img, [(self.n, self.n), (self.n, self.n)], mode='constant', constant_values=0)
        return img

    def resize(self, check=False):
        if check:
            im = Image.open(self.img)
            self.img = np.array(im)
            im.close()
        if self.original_image:
            img = self.read_to_black_white_img()
        else:
            img = self.crop()
        #     img = self.read_to_black_white_img()
        # img = self.convert_to_black_background()
        return cv2.resize(img, (self.dim, self.dim))

    def transform(self):
        t = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Resize(self.dim),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        img = Image.fromarray(self.resize())
        result = torch.autograd.Variable(t(img).unsqueeze(0))
        return result, img


class TransformAll:

    def __init__(self, imgs, original_image, kernel_size, min_thresh, n, dim):
        self.imgs = imgs
        self.kernel_size = kernel_size
        self.min_thresh = min_thresh
        self.n = n
        self.dim = dim
        self.original_image = original_image

    def distribute_collect_images(self):
        images = self.split_numbers()

        img_collection = []

        for image in images:
            trans, im = Process(image, self.original_image, self.kernel_size, self.min_thresh, self.n, self.dim).transform()
            img_collection.append((trans, im))

        return img_collection

    def __sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes

    def split_numbers(self):
        # Read data
        im = Image.open(self.imgs)
        number = np.array(im)
        im.close()
        # number = cv2.imread(img)
        number = cv2.cvtColor(number, cv2.COLOR_BGR2RGB)
        gray_number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)

        # Get threshold and dilation
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        retval, threshold = cv2.threshold(gray_number, self.min_thresh, 255, cv2.THRESH_BINARY_INV)
        dilation = cv2.dilate(threshold, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        left_right_contours, _ = self.__sort_contours(sorted_contours)

        x, y, w, h_max = cv2.boundingRect(sorted_contours[0])
        numbers = []
        for i in range(len(contours)):
            # Get dimensions
            x, y, w, h = cv2.boundingRect(left_right_contours[i])

            if h > h_max / 2:
                # Get digits
                num = number[y:y + h, x:x + w]

                numbers.append(num)

        return numbers
