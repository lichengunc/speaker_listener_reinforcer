"""
This code is used to evaluate the generated sentences.
The language json file is of data [{'ref_id', 'sent'}]
We call corenlp to parse the sentence, then compare it with ground-truth
parses.
The ground-truth parses are saved at 'refer_backup/lib/parse_atts/cache/refcoco/sentToAtts.p'

We will compute precision, recall and harmonic mean for each sentence, 
as well as the overall scores.

The result will be saved in 'ROOT_DIR/cache/parse/dataset/model_idx_split_out.json'
"""
import os
import os.path as osp
import sys
import json
import cPickle as pickle
import argparse
from pprint import pprint
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
ATTS_DIR = osp.join(ROOT_DIR, '../lib/parse_atts/cache')
REFER_DIR = osp.join(ROOT_DIR, 'data')
# load refer API
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'refer_interface'))
from refer import REFER

def load_lang_result(cache_path, dataset_splitBy, model_id, split):
	# load lang_json
	lang_result_path = osp.join(params['cache_path'], 'lang', dataset_splitBy, 'model_id' + model_id + '_' + split)
	Res = json.load(open(lang_result_path + '.json', 'r'))['predictions']  # [{'ref_id', 'sent'}]
	return Res

def sentToParse(Res, num_sents):
	# load corenlp
	sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'corenlp'))
	from corenlp import StanfordCoreNLP
	parser_path = osp.join(ROOT_DIR, 'pyutils', 'corenlp', 'stanford-corenlp-full-2015-01-30')
	stanfordParser = StanfordCoreNLP(parser_path)
	num_sents = len(Res) if num_sents < 0 else num_sents
	print 'stanford parser loaded.'
	# start parsing
	num_sents = len(Res) if num_sents < 0 else num_sents
	for i in range(num_sents):
		ref_id, sent = Res[i]['ref_id'], Res[i]['sent']
		parse = stanfordParser.raw_parse(sent)['sentences'][0]
		Res[i]['parse'] = parse
		print '%s/%s sent is parsed.' % (i+1, num_sents)

def parseToAtts(Res, num_sents, dataset):
	# load attparse
	from attparser.attParser import AttParser
	parser = AttParser(dataset)
	# attribute parsing
	num_sents = len(Res) if num_sents < 0 else num_sents
	for i in range(num_sents):
		ref_id, sent, parse = Res[i]['ref_id'], Res[i]['sent'], Res[i]['parse']
		try:
			parser.reset(parse)
			Res[i]['atts'] = parser.decompose() 
		except:
			Res[i]['atts'] = {'r1': [], 'r2': [], 'r3': [], 'r4': [], 'r5': [], 'r6': [], 'left': []}
		print '%s/%s parse is converted to atts.' % (i+1, num_sents)

def compute_score(Res, num_sents, refer, dataset):
	# sub-functions
	def compute_sc(pred_atts, ref_atts):
		precision = {}
		# compute precision
		for att in pred_atts.keys():
			if len(set(pred_atts[att])) > 0:
				overlap = set(pred_atts[att]) & set(ref_atts[att])
				precision[att] = len(overlap)*1.0/len(set(pred_atts[att]))
		# compute recall
		recall = {}
		for att in ref_atts.keys():
			if len(set(ref_atts[att])) > 0:
				overlap = set(pred_atts[att]) & set(ref_atts[att])
				recall[att] = len(overlap)*1.0/len(set(ref_atts[att]))	
		return precision, recall

	def addtoall(overall, precision, recall):
		# accumulate precision and nump for each att
		for att in precision.keys():
			overall['precision'][att] = overall['precision'].get(att, 0) + precision[att]
			overall['nump'][att] = overall['nump'].get(att, 0) + 1
		# accumulate recall and numr for each att
		for att in recall.keys():
			overall['recall'][att] = overall['recall'].get(att, 0) + recall[att]
			overall['numr'][att] = overall['numr'].get(att, 0) + 1

	# load atts of ground-truth sentences
	sentToAtts = pickle.load(open(osp.join(ATTS_DIR, dataset, 'sentToAtts.p')))

	# for each sent, check with ground-truth sents
	refToEval = {}
	overall = {'precision': {}, 'recall': {}, 'nump': {}, 'numr': {}}
	num_sents = len(Res) if num_sents < 0 else num_sents
	for i in range(num_sents):
		# load atts of pred sent
		ref_id, sent, pred_atts = Res[i]['ref_id'], Res[i]['sent'], Res[i]['atts']
		# load atts of gd sents
		gd_atts = {}
		sent_ids = refer.Refs[ref_id]['sent_ids']
		for sent_id in sent_ids:
			for att in pred_atts.keys():
				gd_atts[att] = gd_atts.get(att, []) + sentToAtts[sent_id][att]
		# compute score
		precision, recall = compute_sc(pred_atts, gd_atts)
		# pprint(pred_atts)
		# pprint(gd_atts)
		# pprint(precision)
		# pprint(recall)
		print '%s/%s parse precision/recall computed.' % (i+1, num_sents)
		# add to overall and refToEval
		refToEval[ref_id] = {'sent': sent, 'atts': pred_atts, 'precision': precision, 'recall': recall}
		addtoall(overall, precision, recall)

	# normalize overall score
	for att in overall['precision']:
		overall['precision'][att] /= float(overall['nump'][att])
		overall['recall'][att] /= float(overall['numr'][att])

	# return
	return overall, refToEval

def run(params):
	# parse
	dataset_splitBy = params['dataset_splitBy']   # in Lua, we simply use dataset denoting detaset_splitBy
	i, j = dataset_splitBy.find('_'), dataset_splitBy.rfind('_')
	dataset = dataset_splitBy[:i]
	splitBy = dataset_splitBy[i+1:] if i == j else dataset_splitBy[i+1:j]
	model_id = params['model_id']
	split = params['split']
	cache_path = params['cache_path']
	num_sents = params['num_sents']

	# load refer
	refer = REFER(REFER_DIR, dataset, splitBy)

	# load lang result
	Res = load_lang_result(cache_path, dataset_splitBy, model_id, split)

	# sentToParse
	sentToParse(Res, num_sents)

	# parseToAtts
	parseToAtts(Res, num_sents, dataset)

	# compute scores
	overall, refToEval = compute_score(Res, num_sents, refer, dataset)
	pprint(overall)

	# write results
	if not osp.isdir(osp.join(cache_path, 'parse', dataset_splitBy)):
		os.mkdir(osp.join(cache_path, 'parse', dataset_splitBy))
	result_path = osp.join(cache_path, 'parse', dataset_splitBy, 'model_id' + model_id + '_' + split)
	with open(result_path + '_out.json', 'w') as outfile:
		json.dump({'overall': overall, 'refToEval': refToEval}, outfile)


if __name__ == '__main__':
	# lang_json is dataset_splitBy_modelidX_split.json
	# get dataset, splitBy, and model id
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_splitBy', default='refcoco_unc', help='name of dataset+splitBy')
	parser.add_argument('--model_id', default='0', help='model_id to be loaded')
	parser.add_argument('--split', default='testA', help='split name, val|test|train')
	parser.add_argument('--cache_path', default='cache', help='cache path')
	parser.add_argument('--num_sents', default=-1, type=int, help='how many sentences to be evaluated. -1 means all')
	args = parser.parse_args()
	params = vars(args)

	# parse
	run(params)
















