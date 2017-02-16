#!/usr/bin/env python
# -*- coding: UTF-8 -*-


__all__ = (
    'generate_ims',
    'make_plate',
)
from ipdb import set_trace
import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
reload(sys)
sys.setdefaultencoding('utf-8')

FONT_HEIGHT = 64 
import common
OUTPUT_SHAPE = common.WINDOW_SHAPE

def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
        else:
                return False

def make_char_ims(font_path, output_height, code):
    font_size = int(output_height * 4)
    
    cnfonts = [f for f in os.listdir(font_path) if f.endswith('cn.ttf')]
    enfonts = [f for f in os.listdir(font_path) if f.endswith('en.ttf')]
    cnfont = ImageFont.truetype(os.path.join(font_path, random.choice(cnfonts)), int(font_size))
    enfont = ImageFont.truetype(os.path.join(font_path, random.choice(enfonts)), int(font_size))

    height = 0.0
    for c in code:
        height = max(height, cnfont.getsize(c)[1] if is_chinese(c) else enfont.getsize(c)[1])

    for c in code:
        font = cnfont if is_chinese(c) else enfont
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def generate_code():
    radio = 0.25
    letters = common.LETTERS.replace(u"O",u"")
    return u"{}{} {}{}{}{}{}".format(
        random.choice(common.PROVINCES),
        random.choice(letters),
        random.choice(letters) if random.random() < radio else random.choice(common.DIGITS),
        random.choice(letters) if random.random() < radio else random.choice(common.DIGITS),
        random.choice(letters) if random.random() < radio else random.choice(common.DIGITS),
        random.choice(letters) if random.random() < radio else random.choice(common.DIGITS),
        random.choice(letters) if random.random() < radio else random.choice(common.DIGITS))
        
def pick_colors():
    first = True
    while first or abs(plate_color - text_color) < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color < plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color

'''
def resize_font(font, ch, size):

    height = font.getsize(ch)[1]
    width = font.getsize(ch)[0]
    
    im = Image.new("RGBA", (width, height), (0, 0, 0))
    
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), ch, (255, 255, 255), font=font)
    
    im = im.resize((size[0], size[1]), Image.ANTIALIAS)

    return numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.
'''    

def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out
   
def make_plate(font_path, code):
    font_height = FONT_HEIGHT 
    text_color, plate_color = pick_colors()   
    char_ims = dict(make_char_ims(font_path, font_height, code)) 

    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(0, 0.1)
    radius = 1 + int(font_height * 0.1 * random.random())

    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
#    print "code:{}".format(code) 
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing
#        print "c:{},len{}".format(c, char_im.shape[1])
        

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
   # scale = 0.8
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-0.8, 0.8) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
  
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds



def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(num_bg_images):
    bg = generate_bg(num_bg_images)
    code = generate_code()
    
    plate, plate_mask, code = make_plate('fonts/', code)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.4,
                            max_scale=0.8,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds
           
def generate_ims(*c):

    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(num_bg_images)

  
if __name__ == "__main__":
    if os.path.exists('test'):
        os.system('rm -rf test')
    os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
                                               "1" if p else "0")
        print fname
        cv2.imwrite(fname, im * 255.)
