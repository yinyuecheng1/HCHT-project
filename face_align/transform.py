import skimage.transform
from load_img import *
import numpy as np
import scipy.ndimage as ndimage

def transform_face(image, eyes):
    """ Your implementation """
    # 1. 计算两只眼睛之间的连线和水平线之间的夹角
    eye0 = np.array(eyes[0])
    eye1 = np.array(eyes[1])
    dist = np.sqrt(np.sum((eye1 - eye0)) ** 2)
    diff = eye1 - eye0
    angle = np.arctan(diff[1] / diff[0])


    # 2.旋转图像
    rot_img = ndimage.rotate(image, np.rad2deg(angle))

    # 3.计算旋转后的图像中眼睛的坐标
    org_eye_center = (eye0 + eye1) / 2
    org_img_center = (np.array(image.shape[:2][::-1]) - 1) / 2. #原图中心坐标
    rot_img_center = (np.array(rot_img.shape[:2][::-1]) - 1) / 2. #旋转后图中心坐标
    org_pos = org_eye_center - org_img_center # 原图中双眼中心和图像中心的向量
    new_pos = np.array([org_pos[0] * np.cos(angle) + org_pos[1] * np.sin(angle),
                        -org_pos[0] * np.sin(angle) + org_pos[1] * np.cos(angle)])
    new_center = new_pos + rot_img_center # 新的双眼中心

    mid_x, mid_y = new_center

    # 4.基于眼睛的坐标来确定人脸框的宽和高（框的坐标）
    MUL = 2.3 # 超参数
    top = int(max(mid_y - MUL * dist, 0))
    bottom = int(min(mid_y + MUL * dist, rot_img.shape[0]))
    left = int(max(mid_x - MUL * dist, 0))
    right = int(min(mid_x + MUL * dist, rot_img.shape[1]))

    # 5. 扣取人脸
    cropped = rot_img[top:bottom, left:right, :]
    transformed = skimage.transform.resize(cropped, [224, 224], mode='constant')

    return transformed

if __name__ == '__main__':
    faces, eyes = load_faces_and_eyes(dir_name='faces_imgs', eye_path='./eyes.pickle')
    transformed_imgs = []
    for i in faces:
        img = faces[i]
        eye = eyes[i]
        transformed = transform_face(img, eye)
        transformed_imgs.append(transformed)

    visualize(transformed_imgs)