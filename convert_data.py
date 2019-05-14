
"""Convert image features from bottom up attention to numpy array"""
import os
import base64
import csv
import sys
import zlib
import json
import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--imgid_list', default='../data/mscoco2014/train/train_ids.txt',
                    help='Path to list of image id')
parser.add_argument('--input_file', default='../data/mscoco2014/train/trainval_resnet101_faster_rcnn_genome_36.tsv',
                    help='tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].')
parser.add_argument('--output_dir', default='../data/mscoco2014/train/',
                    help='Output directory.')
parser.add_argument('--split', default='train',
                    help='train|dev|test')
opt = parser.parse_args()
print(opt)


meta = []
feature = {}
box = {}
for line in open(opt.imgid_list):
    sid = int(line.strip())
    meta.append(sid)
    feature[sid] = None
    box[sid] = None

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

if __name__ == '__main__':
    with open(opt.input_file, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                data = item[field]
                buf = base64.decodestring(data)
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'],-1))
            if item['image_id'] in feature:
                # feature[item['image_id']] = item['features']
                box[item['image_id']] = item['boxes']
    # data_out = np.stack([feature[sid] for sid in meta], axis=0)
    boxes = np.stack([box[sid] for sid in meta], axis=0)
    # print("Final numpy array shape:", data_out.shape)
    print("Final numpy array shape:", boxes.shape)
    # np.save(os.path.join(opt.output_dir, '{}_ims.npy'.format(opt.split)), data_out)
    np.save(os.path.join(opt.output_dir, '{}_boxes.npy'.format(opt.split)), boxes)
