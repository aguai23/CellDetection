# this is an ensemble of all parts of algorithms used for production environment,
# in this file, we put all parts to a set of functions, which is not considered with efficiency
from ctypes import *
import os
def get_patches(input_file, output_path_c,output_path):
    """
       Cut small patches from .kfb image

       Usage:
       all_patches = get_patches

       :param input_file: type is char c like c_char_p(b"/data/Cell/new_kfb/2018-01-15_08_18_23.kfb") orgin kfb image path
       :param output_path_c type is char c like c_char_p(b"/data/Cell/cut_img/2018-01-15_08_18_23/2018-01-15_08_18_23") patch's output path with their name
       :param output_path: patches absolute path

       Output : return the useful patches (file size > 0)
    """
    libc1 = cdll.LoadLibrary("/home/cell/imp/libjpeg.so.9")
    libc = cdll.LoadLibrary("/home/cell/imp/cut.so")
    zoom = 20.0
    fScale = c_float(zoom)
    num_region = 1000000
    XLeft = (c_int * num_region)()
    YLeft = (c_int * num_region)()
    k = 0
    for i in range(0, 150000, 512):
        for j in range(0, 150000, 512):
            XLeft[k] = i
            YLeft[k] = j
            k = k + 1
    X = (c_int * k)()
    Y = (c_int * k)()
    #print(k)
    for i in range(k):
        X[i] = XLeft[i]
        Y[i] = YLeft[i]
    res = libc.ReadImage(input_file, fScale, k, X, Y, 512, 512, output_path_c)
    print("all files:",len(os.listdir(output_path)))
    files = os.listdir(output_path)
    #print(files)
    for file in files:
        file = output_path + file
        if(os.path.getsize(file) == 0):
            os.remove(file)
    print("patch cut ok!")
    return len(os.listdir(output_path))



def read_patches(patch_dir):
    raise NotImplementedError


def get_roi_axis(input_image):
    raise NotImplementedError