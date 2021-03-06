from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import skimage.transform as transform
import sys
import math
from scipy.signal import convolve2d

plt.rcParams['image.cmap'] = 'gray'

###########################################
# User Modifyable Constants
###########################################

# input defaults, necessary for step 0
_images_dir = os.path.join(os.path.expanduser("~"), "programming/comp_photo/mosaic/family/")
_input_images = os.path.join(_images_dir, "input/")
_mosaic_image = os.path.join(_images_dir, "mosaic_image.jpg")

# Necesary starting at step 1
_mosaic_image_scale_factor = 3  # Affects total pixel count of final creation.



# Necessary starting at step 2
# Tuple containing the number of composition images that make up the width by the number of images that create the height
_comp_size = (24, 24) 


_alpha_combining_factor = .5 # Ratio of the final product that should be the mosaic.

###########################################
# Internal Constants
###########################################

_min_dim_filepath = os.path.join(_images_dir, ".min_dimension") # Smallest sidelength of composition photo
_num_images_filepath = os.path.join(_images_dir, ".num_images") # Total number of composition images
_whitelisted_extensions = (".jpg", ".png")
_cropped = os.path.join(_images_dir, "cropped")
_resized = os.path.join(_images_dir, "resized")

#mutable global variables
_mutable_step_count = 0

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

def test_im(im, save=False):
  print(im.shape)
  plt.imshow(im)
  plt.show()
  if save:
    plt.imsave(os.path.join(_images_dir, 'test.jpg'), im)

###########################################
# Cropping
###########################################

def crop_to_square(image):
  def crop_helper(im):
    # Assumes that the number of rows is greater than the number of columns
    if im.shape[0] < im.shape[1]:
      print("Screwed the pooch on dimensions. Try transposing.")
      return
    num_row_remove = im.shape[0]-im.shape[1]
    if not num_row_remove%2==0:
      im = im[:-1,:]
    crop_size = num_row_remove//2
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
  counter = 0
  total_min_dimension = float("inf")
  # TODO: Parallelize this cropping
  for f in map(lambda x: os.path.join(input_dir, x), os.listdir(input_dir)):
    if not os.path.isfile(f) or not has_whitelisted_extension(f):
      print("\nIgnoring: "+str(f))
      continue
    im = None
    try:
      im = plt.imread(f)
    except Exception as e:
      print("\nIgnoring: "+str(f)+" due to "+str(e))
      continue
    cropped_im = crop_to_square(im)
    counter += 1
    total_min_dimension = min(total_min_dimension, cropped_im.shape[0])
    write_image_to_file(cropped_im, output_dir, counter)
    print_update("Finished cropping "+str(counter)+" image(s).")
  write_string_to_file(str(total_min_dimension), _min_dim_filepath)
  write_string_to_file(str(counter), _num_images_filepath)
  print_update_end()

###########################################
# Resizing
###########################################

# Assumes all input images are cropped to squares
def resize_all(input_dir, output_dir, mosaic_file, mosaic_resized_file, comp_size):
  ensure_dir(output_dir)
  min_dim = read_int_from_file(_min_dim_filepath)
  num_im = read_int_from_file(_num_images_filepath)

  # Resize the mosaic image so that the composition images fit evenly by pixels
  composition_length = comp_size[0]
  mosaic_im = plt.imread(mosaic_file)
  mosaic_cropped = crop_to_square(mosaic_im)
  if not mosaic_cropped.shape[0]%composition_length == 0:
    margin = (mosaic_cropped.shape[0]%composition_length)//2
    mod = (mosaic_cropped.shape[0]%composition_length)%2
    mosaic_cropped = crop_margin(mosaic_cropped, margin, margin+mod)
  # Apply scale factor to mosaic image
  mosaic_cropped = transform.rescale(mosaic_cropped, (_mosaic_image_scale_factor, _mosaic_image_scale_factor, 1))
  if mosaic_cropped.shape[0]/composition_length > min_dim:
    raise("Error!!! Not expected to have a mosaic image this large in comparison to the composition images")
    return
  plt.imsave(mosaic_resized_file, mosaic_cropped)

  # Resize the composition_images
  # TODO: parallelize
  sidelength = mosaic_cropped.shape[0]/composition_length
  for x in range(1,num_im+1):
    im = plt.imread(os.path.join(input_dir,"image_"+str(x).zfill(4)+".jpg"))
    resized = transform.resize(im, (sidelength,sidelength,3))
    write_image_to_file(resized, output_dir, x)
    print_update("Finished resizing "+str(x)+" image(s).")
  print_update_end()

###########################################
# Determine Mosaic Composition
###########################################

def gaussian2d(shape=(5,5), sigma=.8):
  m,n = [(ss-1.)/2. for ss in shape]
  x,y = np.ogrid[-m:m+1,-n:n+1]
  f = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
  return f/f.sum()

def evaluation_function(im1,im2):
  # TODO: Use a more in depth evaluation than SSD
  return np.sum((im1-im2)**2)

def featurize(im, comp_height):
  return np.hstack(np.vsplit(im, comp_height))

def defeaturize(im, comp_width):
  return np.vstack(np.hsplit(im, comp_width))

def arrange_composition_photos(mosaic_file, input_dir, output_composition_file, comp_size):
  num_im = read_int_from_file(_num_images_filepath)
  comp_width = comp_size[0]
  comp_height = comp_size[1]
  mosaic_im = skio.imread(mosaic_file)/255.

  # Returns a list of indices in the order that they should be greedily applied to the composition
  def determine_greedy_index_order():
    # Convert mosaic to greyscale for blurring and edge detection (dowsized for ease)
    grey = np.mean(mosaic_im, axis=2)
    small = transform.rescale(grey, min([1, 255./grey.shape[0]])) # downsizes to reoughly 255 pixels
    small_smoothed = convolve2d(small, gaussian2d(), "same")
    high_freq = transform.resize(small-small_smoothed, grey.shape)*255.
    # returns the index-sorted inverse of the sum of high_freq values per composition image
    return np.argsort(list(map(lambda x: 1./np.sum(x), np.hsplit(featurize(high_freq, comp_height), comp_width*comp_height))))

  def determine_greedy_comp_photo_order():
    # resizes mosaic image (for ease of computation) and featurizes
    resized_mosaic_im = mosaic_im
    if _mosaic_image_scale_factor > 1:
      resized_mosaic_im = transform.rescale(resized_mosaic_im, (1./_mosaic_image_scale_factor, 1./_mosaic_image_scale_factor, 1))
    featurized_mosaic_im = featurize(resized_mosaic_im, comp_height)
    sl = featurized_mosaic_im.shape[0]
    # loads resized composition images to memory
    composition_images = {}
    for x in range(1,num_im+1):
      im = skio.imread(os.path.join(input_dir,"image_"+str(x).zfill(4)+".jpg"))/255.
      composition_images[x] = transform.resize(im, (sl,sl,3))

    # determines which composition image corresponds to each part of the mosaic image 
    comp_order = np.zeros(comp_width*comp_height)
    for i in determine_greedy_index_order():
      mosaic_slice = featurized_mosaic_im[:,sl*i:sl*(i+1)]
      d = {k: evaluation_function(mosaic_slice,v) for k, v in composition_images.items()}
      best_comp_im = min(d,key=d.get)
      comp_order[i] = best_comp_im
      del composition_images[best_comp_im]
    return comp_order

  order = determine_greedy_comp_photo_order()
  combined = np.hstack(np.array(list(map(lambda x: skio.imread(os.path.join(input_dir,"image_"+str(int(x)).zfill(4)+".jpg"))/255., order))))
  plt.imsave(output_composition_file, defeaturize(combined, comp_width))

###########################################
# Combining
###########################################

def combine_images(im1, im2, alpha=_alpha_combining_factor):
  assert(alpha >= 0 and alpha <= 1)
  return np.clip((im1*alpha)+(im2*(1-alpha)),0,1)

def combine_mosaic(mosaic_file, input_composition_file, output_file):
  mosaic_im = combine_images(skio.imread(mosaic_file)/255., skio.imread(input_composition_file)/255.)
  plt.imsave(output_file, mosaic_im)


###########################################
# Executing
###########################################

def do_step(func, start_step, *args):
  global _mutable_step_count
  _mutable_step_count += 1
  if (start_step >= _mutable_step_count):
    print("Skipping step " + str(_mutable_step_count-1))
    return
  print("Executing step " + str(_mutable_step_count-1))
  func(*args)

def create_mosaic(start_step=0, input_images_dir=_input_images, mosaic_image=_mosaic_image):
  # Step 0: Crop composition images.
  do_step(crop_composition_images, start_step, input_images_dir, _cropped)
  # Step 1: Resize images
  resized_mosaic_file = os.path.splitext(mosaic_image)[0]+"_resized.jpg"
  do_step(resize_all, start_step, _cropped, _resized, mosaic_image, resized_mosaic_file, _comp_size)
  # Step 2: Arrange composition photos
  arranged_file = os.path.splitext(mosaic_image)[0]+"_arranged.jpg"
  do_step(arrange_composition_photos, start_step, resized_mosaic_file, _resized, arranged_file, _comp_size)
  # Step 3: Combine mosaic and composition photos
  do_step(combine_mosaic, start_step, resized_mosaic_file, arranged_file, os.path.splitext(mosaic_image)[0]+"_composition.jpg")

if __name__ == "__main__":
  create_mosaic(start_step=2)



