import os.path as osp
import os
import sys
import json
import argparse

"""
python scripts/visualize_lang.py --dataset_splitBy xxx --model_id xxx --split xx --beam_size xx
1) load cache/lang/dataset_splitBy/model_idx_split_out.json, which is {ref_id: {'sent', 'Bleu1', 'Bleu2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDER'}}
2) load refer
3) write html and save to vis/lang/dataset
"""
# input
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_unc', help='name of dataset+splitBy')
parser.add_argument('--model_id', default=0, help='model_id to be loaded')
parser.add_argument('--split', default='testA', help='split name, val|test|train')
parser.add_argument('--beam_size', default='1', help='beam size')
args = parser.parse_args()
params = vars(args)

# convert params
dataset_splitBy = params['dataset_splitBy']   # in Lua, we simply use dataset denoting detaset_splitBy
i = dataset_splitBy.find('_')
dataset = dataset_splitBy[:i]
splitBy = dataset_splitBy[i+1:]
model_id = params['model_id']
split = params['split']

# load refer
ROOT_DIR = './'
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'refer'))
REFER_DIR = osp.join(ROOT_DIR, 'data')
from refer import REFER
refer = REFER(REFER_DIR, dataset, splitBy)

# load testing results {ref_id: {'sent', 'Bleu1', 'Bleu2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDER'}}
evaluation = json.load(open(osp.join('cache/lang', dataset_splitBy, str(model_id)+'_'+split+'_beam'+params['beam_size']+'_out.json'), 'r'))
refToEval = evaluation['refToEval']
ref_ids = [int(ref_id) for ref_id in refToEval.keys()]  # keys in json can only be string...suck!
refToPred = {int(ref_id): Eval['sent'] for ref_id, Eval in refToEval.items()}
refToScs = {int(ref_id): Eval for ref_id, Eval in refToEval.items()} # we pretend it doesn't have sent
refToGts = {}
for ref_id in ref_ids:
	sentences = refer.Refs[ref_id]['sentences']
	refToGts[ref_id] = [sent['sent'] for sent in sentences]
refToFilenames = {ref_id: refer.Refs[ref_id]['file_name'] for ref_id in ref_ids}
refToImageId = {ref_id: refer.Refs[ref_id]['image_id'] for ref_id in ref_ids}
imageToRefIds = {}
for ref_id in ref_ids:
	image_id = refer.Refs[ref_id]['image_id']
	imageToRefIds[image_id] = imageToRefIds.get(image_id, []) + [ref_id]
image_ids = imageToRefIds.keys()
image_ids.sort()

# load overall scores
overall = evaluation['overall']

# write html
if not osp.isdir('cache/vis'):
	os.mkdir('cache/vis')
if not osp.isdir('cache/vis/lang'):
	os.mkdir('cache/vis/lang')
vis_folder = osp.join('cache/vis/lang', dataset_splitBy)
if not osp.isdir(vis_folder):
	os.mkdir(vis_folder)
html_path = osp.join('cache/vis/lang', dataset_splitBy, str(model_id)+'_'+split+'_beam'+params['beam_size'])

if dataset == 'refclef':
	url_root = 'http://tlberg.cs.unc.edu/vicente/refimages/'
elif dataset in ['refcoco', 'refcoco+']:
	url_root = 'http://tlberg.cs.unc.edu/vicente/game-code-general/admin/imgs/'
else:  # refcocoG
	url_root = 'http://tlberg.cs.unc.edu/licheng/referit/refcocoG_imgs/'

metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
num_per_page = 100
for page_id, s in enumerate(range(0, len(image_ids), num_per_page)):
	html = open(html_path+'_'+str(page_id)+'.html', 'w')
	html.write('<html><head><style>.sm{color:#009;font-size:0.8em}</style></head>')
	html.write('<body><table border>')
	html.write('<h2>Bleu1  = %.4f</h2>' % overall['Bleu_1'])
	html.write('<h2>Bleu2  = %.4f</h2>' % overall['Bleu_2'])
	html.write('<h2>Bleu3  = %.4f</h2>' % overall['Bleu_3'])
	html.write('<h2>METEOR = %.4f</h2>' % overall['METEOR'])
	html.write('<h2>ROUGE  = %.4f</h2>' % overall['ROUGE_L'])
	html.write('<h2>CIDER  = %.4f</h2>' % overall['CIDEr'])
	html.write('<table style="width:100%"><tr>')
	cnt = 0
	for j in range(s, min(s+num_per_page, len(image_ids))):
		image_id = image_ids[j]
		for ref_id in imageToRefIds[image_id]:
			pred   = refToPred[ref_id]
			gts    = refToGts[ref_id]  # list of sent
			img_url= url_root + refToFilenames[ref_id]
			Bleu1  = refToScs[ref_id]['Bleu_1']
			Bleu2  = refToScs[ref_id]['Bleu_2']

			html.write('<td style="width: 22%">')
			html.write('<table style="width:100%; border:1px solid #ccc; font-size:0.750em; float:left; border-collapse: collapse;">')
			html.write('<tr style="border:1px solid black;">')
			html.write('<td><img src=%s width="180"></td>' % img_url)
			html.write('<td>')
			html.write('<p style="line-height: 100%%;">ref_id%s, image_id%s</p>' % (ref_id, image_id))
			for k, gt in enumerate(gts):
				html.write('<p style="line-height: 100%%;">gd%s: %s</p>' % (k, gt))
			html.write('<p style="line-height: 100%%;">pred: %s</p>' % pred)
			html.write('<p style="line-height: 100%%;">b1: %.3f, b2: %.3f' % (Bleu1, Bleu2))
			html.write('</td></tr>')
			html.write('</table></td>')
			cnt += 1
			if cnt % 4 == 0:
				html.write('</tr><tr>')

	html.write('</tr></table>')
	html.write('</body></html>')
	print 'Page %s written.' % (page_id)

# write another index.html for refering these pages
html = open(html_path+'_main.html', 'w')
html.write('<html><head><style>.sm{color:#009;font-size:0.8em}</style></head>')
html.write('<body><table border>')
html.write('<h2>Bleu1  = %.4f</h2>' % overall['Bleu_1'])
html.write('<h2>Bleu2  = %.4f</h2>' % overall['Bleu_2'])
html.write('<h2>Bleu3  = %.4f</h2>' % overall['Bleu_3'])
html.write('<h2>METEOR = %.4f</h2>' % overall['METEOR'])
html.write('<h2>ROUGE  = %.4f</h2>' % overall['ROUGE_L'])
html.write('<h2>CIDER  = %.4f</h2>' % overall['CIDEr'])
html.write('<ul>')
for page_id, s in enumerate(range(0,len(image_ids), num_per_page)):
	page_html = str(model_id)+'_'+split+'_beam'+params['beam_size']+'_'+str(page_id)+'.html'
	print page_html
	html.write('<li><a target="_blank" href="%s"> page_id%s</a></li>' % (page_html, page_id))
html.write('</ul>')

