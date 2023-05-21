import cv2
import numpy as np
import glob, os
import imutils
import argparse

#import my functions
from utils import pansharpening, align_images



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the folder with input images that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to the folder with template pan-image")
args = vars(ap.parse_args())



image_path = args['images']
template_path = args['template']
#resulting folder
result_path = "results"

image_files = []
template_files = []

for filename in glob.glob(os.path.join(image_path, '*.jpg')):
   image_files.append(filename)


for filename in glob.glob(os.path.join(template_path, '*.jpg')):
   template_files.append(filename)

template_file = template_files[0]
template = cv2.imread(template_file)


for filename in image_files:
   image = cv2.imread(filename)
   #first align image with template
   aligned_image = align_images(image, template)
   #then pan-sharpen image using Esri algorithm, as the best obtained
   pan_image = pansharpening(aligned_image, template, method='esri')
   if not os.path.exists(result_path):
       os.mkdir(result_path)
   name = filename.split("\\")[1]
   #save resulting images
   cv2.imwrite(result_path+f'/pan_{name}', pan_image)




