require 'torch'
require 'nn'
require 'nngraph'
-- exotic import
require 'loadcaffe'
-- local imports
require 'misc.optim_updates'
require 'misc.DataLoader'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local eval_utils = require 'misc.eval_utils'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate Referring Expression Comprehension')
cmd:text()
-- Input paths
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset+splitBy')
cmd:option('-id', '', 'model id to be evaluated')
cmd:option('-mode', 0, '0: use lm, 1: use embedding, 2: ensemble')
cmd:option('-lambda', 0.2, 'weight on lm for ensemble')
-- Test on what split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- misc
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()
-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
-- For CPU
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
-- For GPU
if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
print(opt)
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5   = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_json = data_json, data_h5 = data_h5}
-- also load extracted features: call scripts/extract_xxx_feats before training!
local feats_dir = 'cache/feats/' .. opt.dataset
local featsOpt = {  ann = feats_dir .. '/ann_feats.h5',
					img = feats_dir .. '/img_feats.h5',
					det = feats_dir .. '/det_feats.h5',
					window2 = feats_dir .. '/window2_feats.h5',
					window3 = feats_dir .. '/window3_feats.h5',
					window4 = feats_dir .. '/window4_feats.h5',
					window5 = feats_dir .. '/window5_feats.h5' }
loader:loadFeats(featsOpt)
-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
local model_path = path.join('models/vl_metric_models', opt.dataset, 'model_id' .. opt.id .. '.t7')
local checkpoint = torch.load(model_path)
local protos = checkpoint.protos

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'use_context', 'use_ann', 'use_location', 'margin', 'dif_ann', 'dif_location', 'dif_num',
			   'dif_source', 'dif_pool'}
for k, v in pairs(fetch) do opt[v] = checkpoint.opt[v] end
-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
-- set criterion
local bce_crit = nn.BCECriterion()

-- set mode
for k, v in pairs(protos) do
	print('protos has ' .. k)
	if opt.gpuid >= 0 then v:cuda() end  -- ship to GPU
	bce_crit:cuda()
	v:evaluate()  -- set evalute mode
end

-- initialize
loader:resetImageIterator(opt.split)
local n = 0
local loss_sum = 0
local loss_evals = 0
local accuracy = 0
local predictions = 0


-- evaluate
while true do

	-- fetch data for one image
	local data = loader:getImageBatch(opt.split, opt)
	local image_id = data.image_id
	local img_ann_ids = data.img_ann_ids
	local sent_ids = data.sent_ids	
	local gd_ixs = data.gd_ixs
	assert(#gd_ixs == #sent_ids)
	local feats = data.feats  		-- {(num_anns, dim), ...}
	local zseq = data.zseq 			-- (seq_length, num_sents)
	assert(feats[1]:size(1) == #img_ann_ids)
	assert(zseq:size(2) == #sent_ids)

	-- ship to GPU
	if opt.gpuid >= 0 then
		for k = 1, #feats do feats[k] = feats[k]:cuda() end
		zseq = zseq:cuda()
	end

	-- check over each sent
	local seq_length = loader:getSeqLength()
	for i, sent_id in ipairs(sent_ids) do

		-- expand sent_i's seq
		local sent_zseq = zseq[{ {}, {i} }]:expand(seq_length, #img_ann_ids)
			
		-- labels
		local gd_ix = gd_ixs[i]
		local labels = torch.zeros(#img_ann_ids)
		labels[gd_ix] = 1
		if opt.gpuid >= 0 then labels = labels:cuda() end

		-- forward
		local vis_enc_feats  = protos.vis_encoder:forward(feats) 
		local lang_enc_feats = protos.lang_encoder:forward(sent_zseq)
		local score = protos.metric_net:forward{vis_enc_feats, lang_enc_feats}
		local loss = bce_crit:forward(score, labels)
			
		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1
		local _, pos_sc, max_neg_sc = eval_utils.compute_margin_loss(score, gd_ix, 0)
		if pos_sc > max_neg_sc then accuracy = accuracy + 1 end

		-- print
		local ix0 = data.bounds.it_pos_now - 1
		local ix1 = data.bounds.it_max
		print(string.format('evaluating [%s] performance ... %d/%d sent[%d], acc=%.2f%%, (%.4f)',
			opt.split, ix0, ix1, i, accuracy*100/loss_evals, loss_sum/loss_evals))
	end
	if data.bounds.wrapped then break end  -- we've used up images
end

print(string.format('accuracy = %.2f%%', accuracy/loss_evals*100))































