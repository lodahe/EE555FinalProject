from scipy.io import loadmat
import matplotlib.pyplot as plt
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # Visualize ground truth image
    groundTruthMat = loadmat("BSDS500/data/groundTruth/train/2092.mat")
    images = groundTruthMat['groundTruth'][0][0][0][0]
    for i in range(len(images)):
        print("image {num}".format(num=i))
        plt.imshow(images[i])
        plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
