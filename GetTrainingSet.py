'''
    Func:制作训练集
'''

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2

ct_name = ".nii"
mask_name = ".nii"

ct_path = r"E:\Code\MICCAI-LITS2017-master\dataset\train\ct"
seg_path = r"E:\Code\MICCAI-LITS2017-master\dataset\train\seg"
png_path = './png/'

outputImg_path = r".\trainImage"
outputMask_path = r".\trainMask"

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)  #限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9  #黑色背景区域
        return tmp

def crop_ceter(img, croph, cropw):
    #for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height//2 - (croph//2)
    startw = width//2 - (cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]


if __name__ == "__main__":

    for index, file in enumerate(tqdm(os.listdir(ct_path))):

        # 获取每个病例的四个模态及Mask数据
        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)

        # mask_array[mask_array == 1] = 0  # 肿瘤
        # mask_array[mask_array == 2] = 1

        # 对四个模态分别进行标准化,由于它们对比度不同
        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        # ct_array = ct_array.astype(np.float32)
        # ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        ct_crop = ct_array[start_slice - 1:end_slice + 1, :, :]
        mask_crop = mask_array[start_slice:end_slice + 1, :, :]

        # 切片处理,并去掉没有病灶的切片
        for n_slice in range(mask_crop.shape[0]):
            maskImg = mask_crop[n_slice, :, :] * 255
            cv2.imwrite(png_path + "/seg/" + str(index) + "_" + str(n_slice) + ".png", maskImg)

            ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 3), np.float)
            ctImageArray[:, :, 0] = ct_crop[n_slice - 1, :, :]
            ctImageArray[:, :, 1] = ct_crop[n_slice, :, :]
            ctImageArray[:, :, 2] = ct_crop[n_slice + 1, :, :]

            ctImg = ct_crop[n_slice, :, :]
            ctImg = ctImg.astype(np.float)
            cv2.imwrite(png_path + "/ct/" + str(index) + "_" + str(n_slice) + ".png", ctImageArray)

            imagepath = outputImg_path + "\\" + str(index) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path + "\\" + str(index) + "_" + str(n_slice) + ".npy"
            np.save(imagepath, ctImg)  # (160,160,4) np.float dtype('float64')
            np.save(maskpath, maskImg)  # (160, 160) dtype('uint8') 值为0 1 2 4
    print("Done！")
