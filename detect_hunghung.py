import cv2
import numpy as np
import math

def preprocess(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    return img

class convNN:
    def __init__(self, image, kernel, imgPad=None, padding=0, strides=1):
        self.image = image
        self.kernel = kernel
        self.padding = padding
        self.strides = strides

        self.kernel_x = self.kernel.shape[0]
        self.kernel_y = self.kernel.shape[1]
        self.image_x = self.image.shape[0]
        self.image_y = self.image.shape[1]

        self.output_x = math.ceil((self.image_x - self.kernel_x + 2*self.padding)/strides)
        self.output_y = math.ceil((self.image_y - self.kernel_y + 2*self.padding)/strides)
        self.output = np.zeros((self.output_x, self.output_y))
        self.imgPad = imgPad


        
        
    
    def cross_correlation(self):
        self.kernel = np.flipud(np.fliplr(self.kernel))

    def do_padding(self):
        if self.padding != 0:
            self.imgPad = np.zeros((self.image_x + self.padding*2, self.image_y + self.padding*2))
            self.imgPad[int(self.padding):int(-1 * self.padding), int(self.padding):int(-1 * self.padding)] = self.image
        else:
            self.imgPad = self.image

    def convolution(self):
        m_bound = self.image[1]
        n_bound = self.image[0]

        for y in range(self.image_y-self.kernel_y):
            if y % self.strides == 0:
                for x in range(self.image_x-self.kernel_x):
                    if x % self.strides == 0:
                        self.output[x, y] = (self.kernel * self.imgPad[x: x + self.kernel_x, y: y + self.kernel_y]).sum()
        return self.output                    

    
    



def main():
    img_path = "hunghung.jpg"
    gray_img = preprocess(img_path)

    #kernel = np.array([[-1, 1]])
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    model = convNN(gray_img, kernel)
    model.cross_correlation()
    model.do_padding()
    output = model.convolution()

    # edge_detected_image = conv2D(gray_img, kernel)
    cv2.imwrite("hunghung_after.png", output)

if __name__ == '__main__':
    main()