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
    U, V = M.shape
    for i in range(U):
        for j in range(V):
            try:
                angle = theta[i, j]
                if angle < 0:
                    angle += np.pi
                adj1 = 1000
                adj2 = 1000
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
                    M[i, j] = 0
            except IndexError:
                pass
    return M


# Traditional Canny edge detection algorithm
# Only works with grayscale images!
def apply_canny_operator(image, high_quantile=0.85, low_quantile=0.3, var=1):
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
    M = apply_nonmaximal_suppression(M, theta)
    # Double thresholding
    high = np.quantile(M, high_quantile)
    low = np.quantile(M, low_quantile)
    final_edges = -1*np.ones((U, V))
    for i in range(U):
        for j in range(V):
            if M[i, j] >= high:
                final_edges[i, j] = 255
            elif M[i, j] <= low:
                final_edges[i, j] = 0
    # Hysteresis, connecting strong edges
    for i in range(U):
        for j in range(V):
            if final_edges[i, j] != -1:
                continue
            for k, l in get_neighbors():
                if (i+k >= U) or (j+l >= V) or (i+k < 0) or (j+l < 0):
                    continue
                if final_edges[i+k, j+l] == 255:
                    final_edges[i, j] = 255
                    break
            if final_edges[i, j] != 255:
                final_edges[i, j] = 0
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
def apply_color_canny_operator(image, high_quantile=0.85, low_quantile=0.3):
    # Usually some smoothing is applied, but not in this paper
    # Consider applying some smoothing?
    R = np.array(image[:, :, 0], dtype=np.float64)
    G = np.array(image[:, :, 1], dtype=np.float64)
    B = np.array(image[:, :, 2], dtype=np.float64)
    Rx, Ry = find_gradients(R)
    Gx, Gy = find_gradients(G)
    Bx, By = find_gradients(B)
    Cx = (Rx, Gx, Bx)
    Cy = (Ry, Gy, By)
    direction_numer = 2*(np.multiply(Rx, Ry)+np.multiply(Gx, Gy)+np.multiply(Bx, By))
    direction_denom = norm_squared(Cx)-norm_squared(Cy)
    theta = np.arctan2(direction_numer, direction_denom)/2
    m = np.sqrt(np.multiply(norm_squared(Cx), np.square(np.cos(theta))) +
                np.multiply(direction_numer, np.multiply(np.sin(theta), np.cos(theta))) +
                np.multiply(norm_squared(Cy), np.square(np.sin(theta))))
    m = apply_nonmaximal_suppression(m, theta)
    # Double thresholding
    high = np.quantile(m, high_quantile)
    low = np.quantile(m, low_quantile)
    U, V = m.shape
    final_edges = -1 * np.ones((U, V))
    for i in range(U):
        for j in range(V):
            if m[i, j] >= high:
                final_edges[i, j] = 255
            elif m[i, j] <= low:
                final_edges[i, j] = 0
    # Hysteresis, connecting strong edges
    for i in range(U):
        for j in range(V):
            if final_edges[i, j] != -1:
                continue
            for k, l in get_neighbors():
                if (i + k >= U) or (j + l >= V) or (i + k < 0) or (j + l < 0):
                    continue
                if final_edges[i + k, j + l] == 255:
                    final_edges[i, j] = 255
                    break
            if final_edges[i, j] != 255:
                final_edges[i, j] = 0
    return final_edges


def norm_squared(vector, type='L2'):
    if type == 'L2':
        return np.square(vector[0])+np.square(vector[1])+np.square(vector[2])
    elif type == 'Linf':
        return np.max(np.array([np.max(vector[0]), np.max(vector[1]), np.max(vector[2])]))
    else:
        raise NotImplementedError('Cannot find norm of that type')


if __name__ == '__main__':
    #ex = np.array([[Quaternion(1, 1, 1, 1),Quaternion(1,1,1,2)],[Quaternion(1,1,1,1),Quaternion(1,1,1,1)]])
    #print(ex)
    data_dir = "BSDS500/data/images"
    truth_dir = "BSDS500/data/groundTruth"
    image_prefix = "train/8049"
    test_image = plt.imread(os.path.join(data_dir, image_prefix+'.jpg'))
    plt.imshow(test_image)
    plt.show()
    gray_image = rgb2gray(test_image)
    out = apply_canny_operator(gray_image)
    out2 = apply_color_canny_operator(test_image)
    plt.imshow(out, cmap='gray')
    plt.show()
    plt.imshow(out2, cmap='gray')
    plt.show()
    groundTruthMat = loadmat(os.path.join(truth_dir, image_prefix+'.mat'))
    images = groundTruthMat['groundTruth'][0][0][0][0]
    plt.imshow(images[1], cmap='gray')
    plt.show()

