from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import skimage.transform as transform
import sys
import math
from scipy.signal import convolve2d
import cv2

plt.rcParams['image.cmap'] = 'gray'

###########################################
# User Modifyable Constants
###########################################

# Used to specify which step to start with. Default should remain at 0 for the entire mosaic process to occur.
_start_step = 3

# input defaults, necessary for step 0
_images_dir = os.path.join(os.path.expanduser("~"), "programming/comp_photo/mosaic/family/")
_input_images = os.path.join(_images_dir, "input/")
# value to bias toward center cropping, necessary for step 0.
_center_crop_bias_face = 1.05
_center_crop_bias_saliency = 1.3

# Mosaic composition info. Necessary starting at step 1.
_mosaic_image = os.path.join(_images_dir, "PXL_20231223_233426799.jpg")
_mosaic_image_scale_factor = 2  # Affects total pixel count of final creation.
# Tuple containing the number of composition images that make up the height by the number of images that create the width
_comp_size = (24, 32)

# Necessary starting at step 3
_gaussian_filter_scale = 1./51. # Scale (wrt resolution sidelength) of the filter used for sharpening composition arrangement and smoothing the reference mosaic image.
_alpha_combining_factor = .5 # Ratio of the final product that should be the mosaic.

###########################################
# Internal Constants
###########################################

_min_dim_filepath = os.path.join(_images_dir, ".min_dimension") # Smallest sidelength of composition photo
_num_images_filepath = os.path.join(_images_dir, ".num_images") # Total number of composition images
_whitelisted_extensions = (".jpg", ".png", ".heic")
_cropped = os.path.join(_images_dir, "cropped")
_resized = os.path.join(_images_dir, "resized")

#mutable global variables
_mutable_step_count = 0

###########################################
# Model instantiations
###########################################

_saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
_saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_min_face_size_ratio = 1./16.
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

###########################################
# General utility
###########################################

def ensure_dir(directory):
  if not os.path.exists(directory):
    os.mkdir(directory)

def has_whitelisted_extension(filename):
  return any(map(lambda x: x in (os.path.splitext(filename)[-1]).lower(), _whitelisted_extensions))

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

def clamp_val(val, range):
  return max(range[0], min(range[1], val))

def test_im(im, save=''):
  print(im.shape)
  plt.imshow(im, vmin=0, vmax=1)
  plt.show()
  if save:
    plt.imsave(os.path.join(_images_dir, save+'.jpg'), im, vmin=0, vmax=1)
  return im

###########################################
# Cropping
###########################################

def center_crop(im):
  assert im.shape[0] > im.shape[1], "Screwed the pooch on dimensions. Try transposing."
  num_row_remove = im.shape[0]-im.shape[1]
  if not num_row_remove%2==0:
    im = im[:-1,:]
  crop_size = num_row_remove//2
  return im[crop_size:im.shape[0]-crop_size]

# Generalized center_crop. Removes num_lines from a single axis.
def center_crop_along_axis(im, num_lines, axis):
  assert im.shape[axis] > num_lines, "Cannot remove more lines than exist from the specified axis"
  assert axis in [0,1], "The axis input should be valid [0,1]"
  if num_lines <= 0:
    return im
  if not num_lines%2==0:
    if axis:
      im = im[:,:-1]
    else:
      im = im[:-1]
  crop_size = num_lines//2
  if axis:
    return im[:,crop_size:im.shape[1]-crop_size]
  return im[crop_size:im.shape[0]-crop_size]

# Crop out the top.
def top_crop(im):
  assert im.shape[0] > im.shape[1], "Screwed the pooch on dimensions. Try transposing."
  return im[im.shape[0]-im.shape[1]:]

# Crop out the bottom.
def bot_crop(im):
  assert im.shape[0] > im.shape[1], "Screwed the pooch on dimensions. Try transposing."
  return im[:im.shape[1]]

def choose_crop_type_max_saliency(saliency_map):
  total_saliency = np.sum(saliency_map)
  score_to_crop_type_dict = {}
  score_to_crop_type_dict[np.sum(center_crop(saliency_map))*_center_crop_bias_saliency/total_saliency] = center_crop
  score_to_crop_type_dict[np.sum(top_crop(saliency_map))/total_saliency] = top_crop
  score_to_crop_type_dict[np.sum(bot_crop(saliency_map))/total_saliency] = bot_crop
  return score_to_crop_type_dict[max(score_to_crop_type_dict)]

def compute_face_area_within_height_range(faces, height_range, transposed):
  cropped_face_area = 0
  for (x, y, w, h) in faces:
    if transposed:
      left = clamp_val(x, height_range)
      right = clamp_val(x+w, height_range)
      cropped_face_area += (right-left) * h
    else:
      top = clamp_val(y, height_range)
      bottom = clamp_val(y+h, height_range)
      cropped_face_area += w * (bottom-top)
  return cropped_face_area

def choose_crop_type_max_face_area(im, faces, transposed):
  total_face_area = 0
  for (x, y, w, h) in faces:
    total_face_area += w * h
  # Check if center crop is good enough.
  num_row_remove = (im.shape[0]-im.shape[1])//2
  crop_height_range = [num_row_remove, im.shape[0]-num_row_remove]
  if not ((crop_height_range[1]-crop_height_range[0])%2==0):
    crop_height_range[1] -= 1
  center_face_area = compute_face_area_within_height_range(faces, crop_height_range, transposed)
  if center_face_area/total_face_area > 1/_center_crop_bias_face:
    return center_crop
  score_to_crop_type_dict = {}
  score_to_crop_type_dict[center_face_area*_center_crop_bias_face/total_face_area] = center_crop
  score_to_crop_type_dict[compute_face_area_within_height_range(faces, (im.shape[0]-im.shape[1], im.shape[0]), transposed)/total_face_area] = top_crop
  score_to_crop_type_dict[compute_face_area_within_height_range(faces, (0, im.shape[1]), transposed)/total_face_area] = bot_crop
  return score_to_crop_type_dict[max(score_to_crop_type_dict)]

def crop_to_square(image):
  if image.shape[0] == image.shape[1]:
    return image
  # Detect faces.
  gray_im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  faces = _face_cascade.detectMultiScale(gray_im, 1.05, 11, minSize=(250,250))
  # If no faces compute saliency to use instead.
  success = False
  saliency_map = None
  if len(faces) == 0:
    (success, saliency_map) = _saliency.computeSaliency(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert success
  # Transpose if necessary so we always crop a portrait orientation image.
  transposed = False
  if np.argmin(image.shape[:2]) == 0:
    transposed = True
    image = transpose(image)
    if success:
      saliency_map = np.transpose(saliency_map)
  # Choose which of the three crop functions to use to either maximize face area or saliency.
  crop_func = center_crop
  if len(faces) > 0:
    crop_func = choose_crop_type_max_face_area(image, faces, transposed)
  else:
    crop_func = choose_crop_type_max_saliency(saliency_map)
  cropped_im = crop_func(image)
  if transposed:
    cropped_im = transpose(cropped_im)
  """
  # Shows images that don't use center crop to help with debugging.
  if(crop_func != center_crop):
    print(crop_func)
    for (x, y, w, h) in faces:
      cv2.rectangle(gray_im, (x, y), (x+w, y+h), (255, 0, 0), 20)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(gray_im)
    axarr[1].imshow(cropped_im)
    plt.show()
  """
  return cropped_im

def crop_margin(image, margin1, margin2, margin3, margin4):
  return image[margin1:image.shape[0]-margin2, margin3:image.shape[1]-margin4]

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
      # It's important to use skio.imread here instead of plt.imread since the plt version
      # doesn't always properly account for orientation of pictures.
      im = skio.imread(f)
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
  composition_height = comp_size[0]
  composition_width = comp_size[1]
  mosaic_im = skio.imread(mosaic_file)

  # Resize the mosaic image so that the composition images fit evenly by pixels.
  if composition_width == composition_height:
    mosaic_im = crop_to_square(mosaic_im)
    if not mosaic_im.shape[0]%composition_height == 0:
      margin = (mosaic_im.shape[0]%composition_height)//2
      mod = (mosaic_im.shape[0]%composition_height)%2
      mosaic_im = crop_margin(mosaic_im, margin, margin+mod, margin, margin+mod)
  else:
    # First make the mosaic image match the composition ratio.
    smaller_axis = np.argmin(mosaic_im.shape[:-1])
    larger_axis = int(not smaller_axis)
    assert ((composition_height > composition_width) == bool(smaller_axis)), "Messed up the _comp_size based on the mosaic resolution. Try swapping the values."
    desired_resolution_ratio = comp_size[larger_axis]/comp_size[smaller_axis]
    axis_too_large = larger_axis if mosaic_im.shape[larger_axis]/mosaic_im.shape[smaller_axis] > desired_resolution_ratio else smaller_axis
    num_remove_lines = int(mosaic_im.shape[larger_axis]-(mosaic_im.shape[smaller_axis]*desired_resolution_ratio))
    if num_remove_lines < 0:
      off_axis_remove_lines = abs(num_remove_lines)%desired_resolution_ratio
      mosaic_im = center_crop_along_axis(mosaic_im, off_axis_remove_lines, int(not axis_too_large))
      num_remove_lines = math.ceil(abs(num_remove_lines)/desired_resolution_ratio)
    mosaic_im = center_crop_along_axis(mosaic_im, num_remove_lines, axis_too_large)
    assert math.isclose(mosaic_im.shape[0]/mosaic_im.shape[1], comp_size[0]/comp_size[1], rel_tol=1e-3), "Dimensions of the cropped mosaic " + str(mosaic_im.shape) + " don't match the composition"
    # Now the mosaic image matches the composition ratio, ensure that it fits the compositions evenly by pixels.
    height_margin = (mosaic_im.shape[0]%composition_height)//2
    height_mod = (mosaic_im.shape[0]%composition_height)%2
    width_margin = (mosaic_im.shape[1]%composition_width)//2
    width_mod = (mosaic_im.shape[1]%composition_width)%2
    mosaic_im = crop_margin(mosaic_im, height_margin, height_margin+height_mod, width_margin, width_margin+width_mod)

  # Apply scale factor to mosaic image
  mosaic_im = transform.rescale(mosaic_im, (_mosaic_image_scale_factor, _mosaic_image_scale_factor, 1))
  if mosaic_im.shape[0]/composition_height > min_dim or mosaic_im.shape[1]/composition_width > min_dim:
    raise("Error!!! Not expected to have a mosaic image this large in comparison to the composition images")
    return
  plt.imsave(mosaic_resized_file, mosaic_im)

  # Resize the composition_images
  # TODO: parallelize
  sidelength = mosaic_im.shape[0]/composition_height
  for x in range(1,num_im+1):
    im = skio.imread(os.path.join(input_dir,"image_"+str(x).zfill(4)+".jpg"))
    resized = transform.resize(im, (sidelength,sidelength,3))
    write_image_to_file(resized, output_dir, x)
    print_update("Finished resizing "+str(x)+" image(s).")
  print_update_end()

###########################################
# Determine Mosaic Composition Arrangement
###########################################

def gaussian2d(shape=(5,5), sigma=.8):
  assert len(shape) == 2
  m,n = [(ss-1.)/2. for ss in shape]
  x,y = np.ogrid[-m:m+1,-n:n+1]
  f = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
  return f/f.sum()

def normalized_gaussian2d(shape=(5,5), sigma=.8):
  assert len(shape) == 2
  g = gaussian2d(shape, sigma)
  max_val = g[shape[0]//2, shape[1]//2]
  assert max_val <= 1
  return g/max_val

def evaluation_function(im1,im2):
  # TODO: Use a more in depth evaluation than SSD
  return np.sum((im1-im2)**2)

def featurize(im, comp_height):
  return np.hstack(np.vsplit(im, comp_height))

def defeaturize(im, comp_height):
  return np.vstack(np.hsplit(im, comp_height))

def arrange_composition_photos(mosaic_file, input_dir, output_arrangement_file, comp_size):
  num_im = read_int_from_file(_num_images_filepath)
  comp_height = comp_size[0]
  comp_width = comp_size[1]
  mosaic_im = skio.imread(mosaic_file)/255.

  # Returns a list of indices in the order that they should be greedily applied to the composition
  def determine_greedy_index_order():
    # Convert mosaic to greyscale for blurring and edge detection (dowsized for ease)
    grey = np.mean(mosaic_im, axis=2)
    small = transform.rescale(grey, min([1, 255./grey.shape[0]])) # downsizes to roughly 255 pixels
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
  plt.imsave(output_arrangement_file, defeaturize(combined, comp_height))

###########################################
# Combining
###########################################

def combine_images(im1, im2, alpha=_alpha_combining_factor):
  assert(alpha >= 0 and alpha <= 1)
  return np.clip((im1*alpha)+(im2*(1-alpha)),0,1)

def combine_images_using_mask(im1, im2, mask):
  assert im1.shape == im2.shape
  assert im1.shape == mask.shape
  return np.clip((im1*mask)+(im2*(1-mask)),0,1)


def tint_arrangement(arrangement_im, mosaic_im, comp_size):
  def get_median_color(im):
    return np.median(np.median(im, axis=0), axis=0)
  comp_height = comp_size[0]
  comp_width = comp_size[1]
  featurized_tinted = featurize(arrangement_im, comp_height)
  featurized_mosaic_im = featurize(mosaic_im, comp_height)
  sl = featurized_mosaic_im.shape[0]
  column = 0
  while column < featurized_mosaic_im.shape[1]:
    c = get_median_color(featurized_mosaic_im[:,column:column+sl])
    featurized_tinted[:,column:column+sl] = combine_images(featurized_tinted[:,column:column+sl], np.full((sl,sl,3), get_median_color(featurized_mosaic_im[:,column:column+sl])), alpha=.5)
    column += sl
  return defeaturize(featurized_tinted, comp_height)

def compute_gaussian_face_mask(im):
  assert len(im.shape) == 3
  gray_im = cv2.cvtColor((im*255).astype('u1'), cv2.COLOR_RGB2GRAY)
  output_mask = np.zeros(gray_im.shape)
  min_face_size = int(max(gray_im.shape)*_min_face_size_ratio)
  faces = _face_cascade.detectMultiScale(gray_im, 1.01, 25, minSize=(min_face_size,min_face_size))
  for (x, y, w, h) in faces:
    stacked = np.stack((output_mask[y:y+w, x:x+w], normalized_gaussian2d((w,h), max(w,h)//4)), axis=2)
    output_mask[y:y+w, x:x+w] = np.max(stacked, axis=2)
  return output_mask

def combine_mosaic(input_composition_filename, reference_mosaic_filename, comp_size, tint_filename, smoothed_mosaic_ref_filename, output_filename):
  arrangement_im = skio.imread(input_composition_filename)/255.
  mosaic_im = skio.imread(reference_mosaic_filename)/255.
  assert(arrangement_im.shape == mosaic_im.shape)
  mask = compute_gaussian_face_mask(mosaic_im)
  # Tint the composition arrangement and sharpen it.
  tinted = tint_arrangement(arrangement_im, mosaic_im, comp_size)
  gaussian_sidelength = int(min(mosaic_im.shape[:-1])*_gaussian_filter_scale)
  if gaussian_sidelength % 2 != 1:
    gaussian_sidelength += 1
  tinted_smoothed = cv2.GaussianBlur(tinted, (gaussian_sidelength,gaussian_sidelength), gaussian_sidelength*5*_gaussian_filter_scale)
  tinted_sharpened = np.clip(tinted + (tinted - tinted_smoothed),0,1)
  plt.imsave(tint_filename, tinted_sharpened)
  # Take low frequency of the mosaic reference image (except for faces), then combine with arrangement.
  mask = compute_gaussian_face_mask(mosaic_im)
  gaussian_sidelength = int(min(mosaic_im.shape[:-1])*_gaussian_filter_scale)
  if gaussian_sidelength % 2 != 1:
    gaussian_sidelength += 1
  reference_smoothed = cv2.GaussianBlur(mosaic_im, (gaussian_sidelength,gaussian_sidelength), gaussian_sidelength*5*_gaussian_filter_scale)
  reference_smoothed = combine_images_using_mask(mosaic_im, reference_smoothed, np.dstack((mask, mask, mask)))
  plt.imsave(smoothed_mosaic_ref_filename, reference_smoothed)
  mosaic_im = combine_images(reference_smoothed, tinted_sharpened, alpha=.5)
  plt.imsave(output_filename, mosaic_im)

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
  # Step 1: Resize images.
  resized_mosaic_file = os.path.splitext(mosaic_image)[0]+"_resized.jpg"
  do_step(resize_all, start_step, _cropped, _resized, mosaic_image, resized_mosaic_file, _comp_size)
  # Step 2: Arrange composition photos.
  arranged_file = os.path.splitext(mosaic_image)[0]+"_arranged.jpg"
  do_step(arrange_composition_photos, start_step, resized_mosaic_file, _resized, arranged_file, _comp_size)
  # Step 3: Combine mosaic and composition photos.
  tinted_filename = os.path.splitext(mosaic_image)[0]+"_tinted.jpg"
  smoothed_mosaic_ref_filename = os.path.splitext(mosaic_image)[0]+"_smoothed.jpg"
  mosaic_filename = os.path.splitext(mosaic_image)[0]+"_composition.jpg"
  do_step(combine_mosaic, start_step, arranged_file, resized_mosaic_file, _comp_size, tinted_filename, smoothed_mosaic_ref_filename, mosaic_filename)
  print("Done! Final output file located at " + mosaic_filename)

if __name__ == "__main__":
  create_mosaic(start_step=3)

  # test_im(np.full((682,512,3), (1.,1.,0)))

  #im = skio.imread(os.path.join(_images_dir, 'test.jpg'))/255.
  #featurized = featurize(im, 5)
  #print(np.argsort(list(map(lambda x: 1./np.sum(x), np.hsplit(featurize(im, 5), 5*5)))))
  #test_im(featurized)

  # im = skio.imread(os.path.join(_images_dir, '20220410_125416.jpg'))
  #test_im(im)
  #cropped_im = crop_to_square(im)
  #test_im(cropped_im, "elk_center_cropped")



  # im = cv2.imread(os.path.join(_images_dir, 'PXL_20220918_170511961.jpg'))
  #im = cv2.imread(os.path.join(_images_dir, '20230917_150408.jpg'))
  # im = cv2.imread(os.path.join(_images_dir, 'PXL_20221120_192744494.MP.jpg'))
  # im = cv2.imread(os.path.join(_images_dir, 'PXL_20230704_203822984.jpg'))
  #im = cv2.imread(os.path.join(_images_dir, 'DSC_9421.jpg'))
  #im = cv2.imread(os.path.join(_images_dir, '_DSC8811.jpg'))

  # test_im(normalized_gaussian2d((250,300), 100))
  # test_im(normalized_gaussian2d(sigma=2))

  im = skio.imread(os.path.join(_images_dir, 'PXL_20231223_233426799_resized.jpg'))
  # im = skio.imread(os.path.join('~/programming/comp_photo/mosaic/lauren', 'PXL_20230501_123916777.PORTRAIT.jpg'))
  # im[500:797,500:797] = np.zeros((297,297,3))
  # test_im(im)
  # test_im(compute_gaussian_face_mask(im), "selfie_good_mask")

  gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
  #min_face_size = int(max(gray_im.shape)*_min_face_size_ratio)
  #faces = _face_cascade.detectMultiScale(gray_im, 1.01, 25, minSize=(min_face_size,min_face_size))
  #print(len(faces))
  #for (x, y, w, h) in faces:
  #    cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 15)
  #test_im(im)
  # test_im(im, "family_faces")
  #crop_to_square(im)
  #test_im(featurize(im, 10), 'test_featurize')

"""
  # Face detection
  gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  faces = _face_cascade.detectMultiScale(gray_im, 1.1, 6, minSize=(200,200))
  print(str(len(faces)) + " faces detected.")
  eyes = ()
  # Draw rectangle around the faces
  for (x, y, w, h) in faces:
      print("face size in pixels: " + str(w) + "," + str(w))
      cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
      
      # Detect and draw eyes within the face box
      gray_roi = gray_im[y:y+h, x:x+w]
      color_roi = im[y:y+h, x:x+w]
      eye_detections = _eye_cascade.detectMultiScale(gray_roi, 1.1, 10)
      if len(eye_detections):
        if len(eyes):
          eyes = np.vstack((eyes, eye_detections))
        else:
          eyes = eye_detections
        print(eyes)
  eye_detections = _eye_cascade.detectMultiScale(gray_im, 1.1, 10)
  for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(im,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  # Display the output
  # cv2.imshow('face detections', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
  cv2.imshow('face detections', im)
  cv2.waitKey()
"""




