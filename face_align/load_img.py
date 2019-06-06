import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle

def visualize(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(3, 3, plt_idx)
        plt.imshow(img, cmap=format)
    plt.show()

def load_faces_and_eyes(dir_name='faces_imgs', eye_path='./eyes.pickle'):
    """ Your implementation """
    faces = dict()
    files = os.listdir(dir_name)
    for filename in files:
        img = plt.imread(dir_name+'/'+filename)
        faces[filename] = img

    with open(eye_path, 'rb') as f:
        eyes = pickle.load(f)

    return faces, eyes

if __name__ == '__main__':
    faces, eyes = load_faces_and_eyes(dir_name='faces_imgs', eye_path='./eyes.pickle')
    visualize(faces.values())