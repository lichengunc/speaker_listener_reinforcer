"""
We read "data/detections/dataset_dets.json" and do pre-processing.
It contains {dets: [{box: [x1, y1, w, h], category_id, image_id}]}.
We will assign det_id and h5_id to each item, 
and output {dets: [det_id, h5_id, category_id, image_id, box]}.

The preprocessed json file will be save at cache/prepro/dataset_splitBy/dets.json
"""
import json
import sys
import os.path as osp
from pprint import pprint
import argparse

# parser argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='refcoco', type=str, help='dataset name: refclef, refcoco, refcoco+, refcocog')
parser.add_argument('--splitBy', default='unc', type=str, help='splitBy: unc, google, berkeley')
parser.add_argument('--source', type=str, help='path to the detection or proposal json file')
args = parser.parse_args()
params = vars(args)

# parse params
dataset, splitBy = params['dataset'], params['splitBy']
print 'Preparing detections for dataset[%s]...' % (dataset+'_'+splitBy)
dets_json = params['source']

# load dets = [{box, category_id, image_id, (score)}]
dets = json.load(open(dets_json))['dets']

# assign det_id and h5_id
det_id = 1
h5_id = 1
image_ids = []
for det in dets:
	det['det_id'] = det_id
	det['h5_id'] = h5_id
	image_ids += [det['image_id']]
	det_id += 1
	h5_id += 1
image_ids = list(set(image_ids))
print 'For dataset [%s], we have %s images and %s detections.' % (dataset, len(image_ids), len(dets))

# save to cache/prepro/dataset_splitBy/dets.json
if not osp.isdir('cache/prepro'):
	os.mkdir('cache/prepro')
if not osp.isdir(osp.join('cache/prepro', dataset+'_'+splitBy)):
	os.mkdir(osp.join('cache/prepro', dataset+'_'+splitBy))
output_json = osp.join('cache', 'prepro', dataset+'_'+splitBy, 'dets.json')
with open(output_json, 'w') as of:
	json.dump({'dets': dets}, of)
print 'dets.json saved in %s' % output_json







