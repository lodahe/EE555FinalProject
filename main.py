"""
Author: Austin Kao
Some code adapted from:
https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import os
import queue
from skimage.metrics import structural_similarity as ssim


def get_gaussian_kernel(size=5, variance=1):
    k = int((size-1)/2)
    normal = 1/(2*np.pi*variance)
    H = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            H[i, j] = normal * np.exp(-((i-k-1)**2+(j-k-1)**2)/(2*variance))
    return H


def find_gradients(image):
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ex = ndimage.filters.convolve(image, Sx)
    Ey = ndimage.filters.convolve(image, Sy)
    return Ex, Ey


def get_neighbors():
    diffs = [-1, 0, 1]
    neighbors = []
    for i in diffs:
        for j in diffs:
            if i == 0 and j == 0:
                continue
            neighbors.append((diffs[i], diffs[j]))
    return neighbors


def apply_nonmaximal_suppression(M, theta):
    nonmaximal_M = M.copy()
    U, V = M.shape
    for i in range(U):
        for j in range(V):
            try:
                angle = theta[i, j]
                if angle < 0:
                    angle += np.pi
                adj1 = 1000000
                adj2 = 1000000
                if np.pi/8 < angle <= 3*np.pi/8:
                    adj1 = M[i+1, j-1]
                    adj2 = M[i-1, j+1]
                elif 3*np.pi/8 < angle <= 5*np.pi/8:
                    adj1 = M[i-1, j]
                    adj2 = M[i+1, j]
                elif 5*np.pi/8 < angle <= 7*np.pi/8:
                    adj1 = M[i-1, j-1]
                    adj2 = M[i+1, j+1]
                else:
                    adj1 = M[i, j - 1]
                    adj2 = M[i, j + 1]
                if M[i, j] < adj1 or M[i, j] < adj2:
                    nonmaximal_M[i, j] = 0
            except IndexError:
                pass
    return nonmaximal_M


def double_threshold(image, high, low):
    U, V = image.shape
    strong = queue.Queue()
    visualization = np.zeros((U, V))
    for i in range(U):
        for j in range(V):
            if (image[i, j] > low):
                visualization[i, j] = 25
            if image[i, j] >= high:
                strong.put((i, j))
                visualization[i, j] = 255
    return strong, visualization


def hysteresis(image, strong, high, low):
    U, V = image.shape
    edges = np.zeros((U, V), dtype=np.uint8)
    neighbors = get_neighbors()
    while not strong.empty():
        i, j = strong.get()
        edges[i, j] = 255
        for k, l in neighbors:
            if (i + k >= U) or (j + l >= V) or (i + k < 0) or (j + l < 0):
                continue
            if (edges[i+k, j+l] == 0) and (image[i+k, j+l] > low) and (image[i+k, j+l] < high):
                strong.put((i+k, j+l))
    return edges


# Traditional Canny edge detection algorithm
# Only works with grayscale images!
def apply_canny_operator(image, high_quantile=0.95, low_quantile=0.3, var=1):
    if type(image) is not np.ndarray:
        raise NotImplementedError('Not an ndarray')
    if image.ndim != 2:
        raise NotImplementedError('Wrong dimensions')
    # Noise reduction, convolution with a 5x5 gaussian kernel
    g = get_gaussian_kernel(variance=var)
    smoothed = ndimage.filters.convolve(image, g)
    # Calculation of gradients
    Ex, Ey = find_gradients(smoothed)
    M = np.sqrt(Ex**2 + Ey**2)
    theta = np.arctan2(Ey, Ex)
    # Non-maximum suppression
    U, V = image.shape
    nonmaximal_M = apply_nonmaximal_suppression(M, theta)
    # Double thresholding
    high = np.quantile(nonmaximal_M, high_quantile)
    low = np.quantile(nonmaximal_M, low_quantile)
    strong, threshold_im = double_threshold(nonmaximal_M, high, low)
    # Hysteresis, connecting strong edges
    final_edges = hysteresis(nonmaximal_M, strong, high, low)
    fig, ax = plt.subplots(1, 6)
    for i in range(6):
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)
    ax[0].imshow(image, cmap='gray')
    ax[0].title.set_text('Original Image')
    ax[1].imshow(smoothed, cmap='gray')
    ax[1].title.set_text('Smoothed Image')
    ax[2].imshow(M, cmap='gray')
    ax[2].title.set_text('Gradient Magnitudes')
    ax[3].imshow(nonmaximal_M, cmap='gray')
    ax[3].title.set_text('Nonmaximal Suppression')
    ax[4].imshow(threshold_im, cmap='gray')
    ax[4].title.set_text('Double Thresholding')
    ax[5].imshow(final_edges, cmap='gray')
    ax[5].title.set_text('Final Result')
    plt.show()
    return final_edges




# Using formula from:
# http://support.ptc.com/help/mathcad/en/index.html#page/PTC_Mathcad_Help/example_grayscale_and_color_in_images.html
def rgb2gray(image):
    if type(image) is not np.ndarray:
        raise NotImplementedError('Not an ndarray')
    if image.ndim != 3 or image.shape[2] != 3:
        raise NotImplementedError('Wrong dimensions')
    return 0.299*image[:, :, 0] + 0.587*image[:, :, 1] + 0.114*image[:, :, 2]


# Canny operator extended to color images
# Paper: https://www.csd.uoc.gr/~hy371/bibliography/EdgeColorImages.pdf
def apply_color_canny_operator(image, high_quantile=0.95, low_quantile=0.3, var=1):
    # Usually some smoothing is applied, but not in this pape
    R = np.array(image[:, :, 0], dtype=np.float64)
    G = np.array(image[:, :, 1], dtype=np.float64)
    B = np.array(image[:, :, 2], dtype=np.float64)
    kernel = get_gaussian_kernel(variance=var)
    smooth_R = ndimage.filters.convolve(R, kernel)
    smooth_G = ndimage.filters.convolve(R, kernel)
    smooth_B = ndimage.filters.convolve(R, kernel)
    Rx, Ry = find_gradients(smooth_R)
    Gx, Gy = find_gradients(smooth_G)
    Bx, By = find_gradients(smooth_B)
    Cx = (Rx, Gx, Bx)
    Cy = (Ry, Gy, By)
    direction_numer = 2*(np.multiply(Rx, Ry)+np.multiply(Gx, Gy)+np.multiply(Bx, By))
    direction_denom = norm_squared(Cx)-norm_squared(Cy)
    theta = np.arctan2(direction_numer, direction_denom)/2
    m = np.sqrt(np.multiply(norm_squared(Cx), np.square(np.cos(theta))) +
                np.multiply(direction_numer, np.multiply(np.sin(theta), np.cos(theta))) +
                np.multiply(norm_squared(Cy), np.square(np.sin(theta))))
    nonmaximal_m = apply_nonmaximal_suppression(m, theta)
    # Double thresholding
    high = np.quantile(nonmaximal_m, high_quantile)
    low = np.quantile(nonmaximal_m, low_quantile)
    strong, threshold_im = double_threshold(nonmaximal_m, high, low)
    # Hysteresis, connecting strong edges
    final_edges = hysteresis(nonmaximal_m, strong, high, low)
    return final_edges


def norm_squared(vector, type='L2'):
    if type == 'L2':
        return np.square(vector[0])+np.square(vector[1])+np.square(vector[2])
    elif type == 'Linf':
        return np.square(np.maximum(np.maximum(vector[0], vector[1]), vector[2]))
    else:
        raise NotImplementedError('Cannot find norm of that type')


if __name__ == '__main__':
    #ex = np.array([[Quaternion(1, 1, 1, 1),Quaternion(1,1,1,2)],[Quaternion(1,1,1,1),Quaternion(1,1,1,1)]])
    #print(ex)
    data_dir = "BSDS500/data/images"
    truth_dir = "BSDS500/data/groundTruth"
    #image_prefix = "train/25098"
    #image_prefix = "val/3096"
    #image_prefix = "train/198023"
    test_image = plt.imread(os.path.join(data_dir, image_prefix + '.jpg'))
    gray_image = rgb2gray(test_image)
    out = apply_canny_operator(gray_image, high_quantile=0.875, low_quantile=0.25, var=25)
    out2 = apply_color_canny_operator(test_image, high_quantile=0.875, low_quantile=0.25, var=25)
    groundTruthMat = loadmat(os.path.join(truth_dir, image_prefix + '.mat'))
    images = groundTruthMat['groundTruth'][0][0][0][0]
    fig, ax = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_visible(False)
    ax[0, 0].imshow(test_image)
    ax[0, 0].title.set_text('Original Image')
    ax[0, 1].imshow(images[1], cmap='gray')
    ax[0, 1].title.set_text('Ground Truth')
    ax[1, 0].imshow(out, cmap='gray')
    ax[1, 0].title.set_text('Traditional Canny')
    ax[1, 1].imshow(out2, cmap='gray')
    ax[1, 1].title.set_text('Canny with Color')
    a = np.sum(images[1] > 0)
    b = np.sum((out > 0) & (images[1] > 0))
    c = np.sum((out2 > 0) & (images[1] > 0))
    print("Traditional Canny edge accuracy: {}".format(b / a))
    print("Canny with color edge accuracy: {}".format(c / a))
    print("SSIM for traditional Canny: {}".format(ssim(out, images[1])))
    print("SSIM for Canny with color: {}".format(ssim(out2, images[1])))
    plt.show()

