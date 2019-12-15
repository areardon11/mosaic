from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import skimage.transform as transform
import sys
import math

###########################################
# Constants
###########################################

# input defaults
_images_dir = os.path.join(os.path.expanduser("~"), "programming/comp_photo/mosaic/relationship/")
_input_images = os.path.join(_images_dir, "input_full")
_mosaic_image = os.path.join(_images_dir, "mosaic_image.jpg")

# tuning variables
_mosaic_image_scale_factor = 5  # Affects total pixel count of final creation.

# script creations
_metadata_dir = os.path.join(os.path.expanduser("~"), "programming/comp_photo/mosaic/code/metadata/")
_min_dim_filename = "min_dimension"
_num_images_filename = "num_images"
_whitelisted_extensions = (".jpg", ".png")
_cropped = os.path.join(_images_dir, "cropped2")
_resized = os.path.join(_images_dir, "resized")

#mutable global variables
_mutable_step_count = 0

#14,53,65,77,121,148

#46,113,121

###########################################
# General utility
###########################################

def ensure_dir(directory):
  if not os.path.exists(directory):
    os.mkdir(directory)

def has_whitelisted_extension(filename):
  return any(map(lambda x: x in os.path.splitext(filename)[-1], _whitelisted_extensions))

def write_image_to_file(image, output_dir, num):
  plt.imsave(os.path.join(output_dir,"image_"+str(num).zfill(4)+".jpg"), image)

def write_string_to_file(data, filepath):
  with open(filepath, "w+") as f:
    f.write(data)

def read_lines_from_file(filepath):
  lines = []
  with open(filepath, "r") as f:
    lines = f.readlines()
  return lines
def read_int_from_file(filepath):
  lines = read_lines_from_file(filepath)
  if not len(lines) == 1:
    print("Error: Expected there to be a single 'int' in "+filepath)
    return
  return int(lines[0])

def print_update(string):
  print(string, end='\r')
  sys.stdout.flush()
def print_update_end():
  print("")

def transpose(im):
  return np.transpose(im, (1, 0, 2))

def test_im(im):
  print(im.shape)
  plt.imshow(im)
  plt.show()

###########################################
# Cropping
###########################################

def crop_to_square(image):
  def crop_helper(im):
    # Assumes that the number of rows is greater than the number of columns
    if im.shape[0] < im.shape[1]:
      print("Screwed the pooch on dimensions. Try transposing.")
      return
    num_col_remove = im.shape[0]-im.shape[1]
    if not num_col_remove%2==0:
      im = im[:,:-1]
    crop_size = num_col_remove/2
    return im[crop_size:im.shape[0]-crop_size]

  if image.shape[0] == image.shape[1]:
    return image
  transposed = False
  min_dim_ind = np.argmin(image.shape[:2])
  if min_dim_ind == 0:
    transposed = True
    image = transpose(image)
  cropped_im = crop_helper(image)
  if transposed:
    cropped_im = transpose(cropped_im)
  return cropped_im

def crop_margin(image, margin1, margin2):
  return image[margin1:image.shape[0]-margin2, margin1:image.shape[1]-margin2]

def crop_composition_images(input_dir, output_dir):
  ensure_dir(output_dir)
  ensure_dir(_metadata_dir)
  counter = 0
  total_min_dimension = float("inf")
  # TODO: Parallelize this cropping
  for f in map(lambda x: os.path.join(input_dir, x), os.listdir(input_dir)):
    if not os.path.isfile(f) or not has_whitelisted_extension(f):
      print("\nIgnoring: "+str(f))
      continue
    im = plt.imread(f)
    cropped_im = crop_to_square(im)
    counter += 1
    total_min_dimension = min(total_min_dimension, cropped_im.shape[0])
    write_image_to_file(cropped_im, output_dir, counter)
    print_update("Finished cropping "+str(counter)+" image(s).")
  write_string_to_file(str(total_min_dimension), os.path.join(_metadata_dir, _min_dim_filename))
  write_string_to_file(str(counter), os.path.join(_metadata_dir, _num_images_filename))
  print_update_end()

###########################################
# Resizing
###########################################

# Assumes all input images are cropped to squares
def resize_all(input_dir, output_dir, mosaic_file, mosaic_resized_file):
  ensure_dir(output_dir)
  min_dim = read_int_from_file(os.path.join(_metadata_dir, _min_dim_filename))
  num_im = read_int_from_file(os.path.join(_metadata_dir, _num_images_filename))

  # Resize the mosaic image so that the composition images fit evenly by pixels
  composition_length = int(math.sqrt(num_im))
  mosaic_im = plt.imread(mosaic_file)
  mosaic_cropped = crop_to_square(mosaic_im)
  # Apply scale factor to mosaic image
  mosaic_cropped = transform.rescale(mosaic_cropped, _mosaic_image_scale_factor)
  if not mosaic_cropped.shape[0]%composition_length == 0:
    margin = (mosaic_cropped.shape[0]%composition_length)//2
    mod = (mosaic_cropped.shape[0]%composition_length)%2
    mosaic_cropped = crop_margin(mosaic_cropped, margin, margin+mod)
  if mosaic_cropped.shape[0]/composition_length > min_dim:
    print("Error!!! Not expected to have a mosaic image this large in comparison to the composition images")
    return
  plt.imsave(mosaic_resized_file, mosaic_cropped)

  # Resize the composition_images
  # TODO: parallelize
  sidelength = mosaic_cropped.shape[0]/composition_length
  counter = 0
  for f in map(lambda x: os.path.join(input_dir, x), os.listdir(input_dir)):
    if not os.path.isfile(f) or not has_whitelisted_extension(f):
      print("\nIgnoring: "+str(f))
      continue
    counter += 1
    im = plt.imread(f)
    resized = transform.resize(im, (sidelength,sidelength,3))
    write_image_to_file(resized, output_dir, counter)
    print_update("Finished resizing "+str(counter)+" image(s).")
  print_update_end()



###########################################
# Executing
###########################################

def do_step(func, start_step, *args):
  global _mutable_step_count
  _mutable_step_count += 1
  if (start_step >= _mutable_step_count):
    return
  print("Executing step " + str(_mutable_step_count-1))
  func(*args)

def create_mosaic(start_step=0, input_images_dir=_input_images, mosaic_image=_mosaic_image):
  # Step 0: Crop composition images.
  do_step(crop_composition_images, start_step, input_images_dir, _cropped)
  # Step 1: Resize images
  resized_mosaic_file = os.path.splitext(mosaic_image)[0]+"_resized.jpg"
  do_step(resize_all, start_step, _cropped, _resized, mosaic_image, resized_mosaic_file)

def get_sample_im():
  return plt.imread(os.path.join(_images_dir, _input_full, "IMG_20180923_135334.jpg"))
  
def test():
  print(read_int_from_file(os.path.join(_metadata_dir, _num_images_filename)))

if __name__ == "__main__":
  create_mosaic(start_step=1)
  # test()

