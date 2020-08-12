#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: ger_video.py
@time: 2020/8/6 16:55
@version 1.0
@desc:

"""

import cv2
from pathlib import Path

input_dir = Path('output/0730_MUNIT_cat2dog/outputs/cat2dog_munit/images')

images = list(input_dir.glob('gen_a2b_test_*'))

output_dir = Path('output/videos')
output_dir.mkdir(exist_ok=True, parents=True)
output_file = 'munit.mp4'

fps = 12
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
frame_size = cv2.imread(str(images[0])).shape[:2]
video_writer = cv2.VideoWriter(str(output_dir / output_file), fourcc,fps, (frame_size[0],frame_size[1]))

for img in images:
    img = cv2.imread(str(img))
    video_writer.write(img)

video_writer.release()
