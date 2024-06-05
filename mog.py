import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt


def show_image(img):
    """
    Shows an image (img) using matplotlib
    """
    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3 or img.shape[-1] == 4:
            plt.imshow(img[..., :3])
        if img.shape[-1] == 1 or img.shape[-1] > 4:
            plt.imshow(img[:, :], cmap="gray")
        plt.show()


def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    new_img = np.zeros(img.shape)

    off_X = int(np.floor(kernel.shape[0] / 2))
    off_Y = int(np.floor(kernel.shape[1] / 2))
    padding = off_X
    if off_X < off_Y:
        padding = off_Y

    padded_Img = np.pad(img, padding, mode="edge")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_kernel = padded_Img[x : x + off_X * 2 + 1, 
                                    y : y + off_Y * 2 + 1]
            calc_new_value = np.multiply(kernel,img_kernel)
            new_img[x][y] = calc_new_value.sum()
    return new_img

def magnitude_of_gradients(RGB_img):
    """
    Computes the magnitude of gradients using x-sobel and y-sobel 2Dconvolution

    :param img: RGB image
    :return: length of the gradient
    """
    mono_img = RGB_img[..., :3]@np.array([0.299, 0.587, 0.114])
    mono_img2 = RGB_img[..., :3]@np.array([0.299, 0.587, 0.114])

    x_sobel = np.matrix('-1.0 0.0 1.0; -2.0 0.0 2.0; -1.0 0.0 1.0')
    y_sobel = np.matrix('1.0 2.0 1.0; 0.0 0.0 0.0; -1.0 -2.0 -1.0')
    soft = np.matrix('.1 .1 .1; .1 0 .1; .1 .1 .1')
    flip = np.matrix('0 0 0; 0 -1 0; 0 0 0')
    sharp = np.matrix('-1 -1 -1 -1 -1;-1 -1 -1 -1 -1;-1 -1 49 -1 -1;-1 -1 -1 -1 -1; -1 -1 -1 -1 -1')

    yGradient = convolution2D(mono_img, y_sobel)
    xGradient = convolution2D(mono_img2, x_sobel)
    softImg = convolution2D(mono_img2, soft)
    flipImg = convolution2D(mono_img2, flip)
    sharpImg = convolution2D(mono_img2, sharp)
    
    mpimage.imsave("./kernel_Images/ySobel.png", yGradient, cmap="gray")
    mpimage.imsave("./kernel_Images/xSobel.png", xGradient, cmap="gray")
    mpimage.imsave("./kernel_Images/soft.png", softImg, cmap="gray")
    mpimage.imsave("./kernel_Images/flip.png", flipImg, cmap="gray")
    mpimage.imsave("./kernel_Images/sharp.png", sharpImg, cmap="gray")

    mgo = np.zeros(mono_img.shape)
    
    for x in range(mono_img.shape[0]):
        for y in range(mono_img.shape[1]):
            mgo[x][y] = (xGradient[x, y] ** 2 + yGradient[x, y] ** 2) ** .5

    return mgo

if __name__ == '__main__':
    # Bild laden und zu float konvertieren
    img = mpimage.imread('bilder/tower.jpg')
    img = img.astype("float64")

    # Wandelt RGB Bild in ein grayscale Bild um
    gray = img[..., :3]@np.array([0.299, 0.587, 0.114])
    
    testImg = magnitude_of_gradients(img)
    mpimage.imsave("./mgo_Images/test.png", testImg, cmap="gray")