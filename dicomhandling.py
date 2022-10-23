from pydicom import dcmread
from pydicom import *
import os
import pydicom.data
import numpy as np
from matplotlib import pyplot, cm
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2  
import imutils
import functools


##########################   TASK 1   #######################
class DcmFilter :
    
    def __init__(self, name_DICOM):
        self.name_DICOM = name_DICOM
        
    def get_image_dicom (self):
        filename = pydicom.data.data_manager.get_files(path,self.name_DICOM)[0]
        self.ds = pydicom.dcmread(filename)

        self.ipp = self.ds[0x0020,0x0032].value
      
        self.Dim_dicom = (int(self.ds.Rows), int(self.ds.Columns))
        
      #  plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
      #  plt.show()
    def store_numpy(self):
        array_dicom = np.zeros(self.Dim_dicom, dtype=self.ds.pixel_array.dtype)
        array_dicom[:, :] = self.ds.pixel_array
        self.original = array_dicom
        
    def Gauss_2D_Dicom (self):
        Image_filt = gaussian_filter(self.original, sigma)
        self.filtered =Image_filt
        
##########################   TASK 2   #######################        
class DcmRotate:
    def __init__(self, name_DICOM_rotate):
        self.name_DICOM_rotate = name_DICOM_rotate
        
    def get_image_dicom_rotate (self):
        filename = pydicom.data.data_manager.get_files(path,self.name_DICOM_rotate)[0]
        self.ds = pydicom.dcmread(filename)

        self.ipp = self.ds[0x0020,0x0032].value
        
        
        self.Dim_dicom = (int(self.ds.Rows), int(self.ds.Columns))
        
       
    def store_numpy_rotate(self):
        array_dicom = np.zeros(self.Dim_dicom, dtype=self.ds.pixel_array.dtype)
        array_dicom[:, :] = self.ds.pixel_array
        self.original = array_dicom

    def rotate_dicom(self):
        self.Rotated_image = imutils.rotate(self.original, angle=angle)
        
##########################   TASK 3   #######################
def check_ipp(list_1_dicom,list_2_dicom, equal_result):
    if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list_1_dicom, list_2_dicom), True) : 
        equal_result = True
        return equal_result
    else :
        return equal_result

    
path = 'C:/Users/laura/AppData/Local/Programs/Python/Python36/Quibim/img'



##########################   TASK 1   #######################
sigma = 3
dcm_filter = DcmFilter('IM-0001-0035-0001.dcm')
dcm_filter.get_image_dicom()
dcm_filter.store_numpy()
dcm_filter.ipp
print(dcm_filter.ipp)
#New_patient.Gauss_2D_Dicom()


##########################   TASK 2   #######################
angle = 270
dcm_rotate = DcmRotate('IM-0001-0086-0001.dcm')
dcm_rotate.get_image_dicom_rotate()
dcm_rotate.store_numpy_rotate()
dcm_rotate.ipp
print(dcm_rotate.ipp)
#New_patient.rotate_dicom()


##########################   TASK 3   #######################
Ipp_equal = False
print(check_ipp(dcm_rotate.ipp,dcm_filter.ipp,Ipp_equal))


