"""
This code is used to evalute the validation results.
The validation json file is of data [{'ref_id', 'sent'}]
We call REFER and RefEvaluation to evalute different types of scores.

Things from RefEvaluation of interests:
evalRefs  - list of ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']
eval      - dict of {metric: score}
refToEval - dict of {ref_id: ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']}

We name output file as 'model_idx_split_out.json'
"""
import os
import os.path as osp
import sys
import json
import argparse

# val_json is dataset_splitBy_modelidX_val|test.json
# get dataset, splitBy, and model json
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco+_unc', help='name of dataset+splitBy')
parser.add_argument('--model_id', default='visdif', help='model name: baseline, baseline_mmi')
parser.add_argument('--beam_size', default='1')
parser.add_argument('--split', default='testA')
parser.add_argument('--write_result', default=-1, help='write result into cache_path')
parser.add_argument('--refer_dir', default='new_data', help='data or new_data, refer directory that stores annotations.')
args = parser.parse_args()
params = vars(args)

# parse 
dataset_splitBy = params['dataset_splitBy']   # in Lua, we simply use dataset denoting detaset_splitBy
i, j = dataset_splitBy.find('_'), dataset_splitBy.rfind('_')
dataset = dataset_splitBy[:i]
splitBy = dataset_splitBy[i+1:] if i == j else dataset_splitBy[i+1:j]
file_name = params['model_id']+'_'+params['split']+'_beam'+params['beam_size']+'.json'
result_path = osp.join('cache', 'lang', dataset_splitBy, file_name)

# load refer and refToEvaluation
sys.path.insert(0, osp.join('pyutils', 'refer'))
from refer import REFER
refer = REFER(params['refer_dir'], dataset, splitBy)

# load predictions
Res = json.load(open(result_path, 'r'))['predictions']  # [{'ref_id', 'sent'}]

# regular evaluate
sys.path.insert(0, osp.join('pyutils', 'refer2', 'evaluation'))
from refEvaluation import RefEvaluation
refEval = RefEvaluation(refer, Res)
refEval.evaluate()
overall = {}
for metric, score in refEval.eval.items():
	overall[metric] = score
print overall

if params['write_result'] > 0:
	refToEval = refEval.refToEval
	for res in Res:
		ref_id, sent = res['ref_id'], res['sent']
		refToEval[ref_id]['sent'] = sent
	with open(result_path[:-5] + '_out.json', 'w') as outfile:
		json.dump({'overall': overall, 'refToEval': refToEval}, outfile)


# cross evaluate
from crossEvaluation import CrossEvaluation
ceval = CrossEvaluation(refer, Res)
ceval.cross_evaluate()
ceval.make_ref_to_evals()
ref_to_evals = ceval.ref_to_evals  # ref_to_evals = {ref_id: {ref_id: {method: score}}}
# compute cross score
ceval.Xscore('CIDEr')











