from pydicom import dcmread
from pydicom import *
import os
import pydicom.data
import numpy as np
from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2  
import imutils
import functools
import sys
import cv2
import matplotlib.image as mpimg
from skimage.color import rgb2gray

class IncorrectNumberOfImages(Exception):
    "Exception raised when number of images different from 2."
    def __init__(self, message="Incorrect number of images. Aborting"):
       
        self.message = message
        super().__init__(self.message)

class SameImagePositionPatient(Exception):
    "Exception raised when images have same position."
    def __init__(self, message="The DICOM files appear to be the same. Aborting."):
       
        self.message = message
        super().__init__(self.message)
        
class DcmFilter :
    
    def __init__(self, name_DICOM):
        self.name_DICOM = name_DICOM
        
    def get_image_dicom (self):
        filename = pydicom.data.data_manager.get_files(path,self.name_DICOM)[0]
        self.ds = pydicom.dcmread(filename)

        self.ipp = self.ds[0x0020,0x0032].value
      
        self.Dim_dicom = (int(self.ds.Rows), int(self.ds.Columns))
       
  
    def store_numpy(self):
        array_dicom = np.zeros(self.Dim_dicom, dtype=self.ds.pixel_array.dtype)
        array_dicom[:, :] = self.ds.pixel_array
        self.original = array_dicom
        
    def Gauss_2D_Dicom (self):
        Image_filt = gaussian_filter(self.original, sigma)
        self.filtered =Image_filt
        
def save_img (Pat_1,Pat_2,path_residue):
    Image_redisue_original = cv2.subtract(Pat_2.original,Pat_1.original)
    
    Image_redisue_filtered = cv2.subtract(Pat_2.filtered,Pat_1.filtered)
   
    img_orig_path = os.path.join(path_residue,'original.jpg')
    img_filtered_path = os.path.join(path_residue,'filtered.jpg')
    
   # cv2.imwrite(img_orig_path, Image_redisue_original*255)
   # cv2.imwrite(img_filtered_path, Image_redisue_filtered)

    mpimg.imsave(img_orig_path,Image_redisue_original)
    mpimg.imsave(img_filtered_path,Image_redisue_filtered)

   # plt.imshow(Image_redisue_original)
   # plt.show()
    
    
def main():
    file_ = os.listdir(path)
    file_dicom = [i for i in file_ if i.endswith('.dcm')]

    if len(file_dicom) !=  2 :
        raise IncorrectNumberOfImages

    
  
       
    Patient_1 = DcmFilter(file_dicom[0])
    Patient_1.get_image_dicom()
    Patient_1.store_numpy()
    Patient_1.Gauss_2D_Dicom()


    Patient_2 = DcmFilter(file_dicom[1])
    Patient_2.get_image_dicom()
    Patient_2.store_numpy()
    Patient_2.Gauss_2D_Dicom()

    if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, Patient_1.ipp, Patient_2.ipp), True) : 
        raise SameImagePositionPatient
    
    path_residue=os.path.join(path,'residues')
    if not os.path.exists(path_residue):
       os.makedirs(path_residue)
       
    save_img(Patient_1,Patient_2,path_residue)

   
    
  
##########For input folder, please remove space before and after '-' , rename folder to T1_3D_TFE-301#####


if __name__ == "__main__":
    sigma = 3
    param_1= sys.argv[1]
    
    path__= os.getcwd()
    path=os.path.join(path__,param_1)
   
    main()
