require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.modules.SplitEmbedding'
require 'misc.modules.SplitGeneration'
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
local model_path = path.join('models', opt.dataset, 'model_id' .. opt.id .. '.t7')
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
-- set mode
for k, v in pairs(protos) do
	print('protos has ' .. k)
	if opt.gpuid >= 0 then v:cuda() end  -- ship to GPU
	v:evaluate()  -- set evalute mode
end

-- initialize
loader:resetImageIterator(opt.split)
local n = 0
local loss_sum = 0
local loss_evals = 0
local accuracy = 0
local predictions = {}

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
	local seqz = data.seqz  		-- (seq_length, num_sents)
	local zseq = data.zseq 			-- (seq_length, num_sents)
	assert(feats[1]:size(1) == #img_ann_ids)
	assert(seqz:size(2) == #sent_ids)

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
		local sent_seqz = seqz[{ {}, {i} }]:expand(seq_length, #img_ann_ids)

		-- forward
		local vis_enc_feats  = protos.vis_encoder:forward(feats)
		local lang_enc_feats = protos.lang_encoder:forward(sent_zseq)
		local cossim, vis_emb_feats = unpack(protos.cca_embedding:forward{vis_enc_feats, lang_enc_feats})
		local vis_feats = protos.vis_combiner:forward{vis_enc_feats, vis_emb_feats}
		local logprobs  = protos.lm:forward{vis_feats, sent_seqz}  -- (seq_length+1, #img_ann_ids, vocab_size+1)

		-- language ranking margin loss
		local lm_scores = -computeLosses(logprobs, sent_seqz):float()  -- (#img_ann_ids, )

		-- embedding ranking margin loss
		local emb_scores = cossim:float()

		-- evaluate using what score?
		local gd_ix = gd_ixs[i]
		local max_ix
		local pos_sc
		local max_neg_sc
		local mode_str 
		assert(opt.mode==0 or opt.mode==1 or opt.mode==2)
		if opt.mode == 0 then
			-- use language model
			_, max_ix = torch.max(lm_scores, 1)
			max_ix = max_ix[1]
			_, pos_sc, max_neg_sc = eval_utils.compute_margin_loss(lm_scores, gd_ix, 0)
			mode_str = 'lm'
		elseif opt.mode == 1 then
			-- use embedding model
			_, max_ix = torch.max(emb_scores, 1)
			max_ix = max_ix[1]
			_, pos_sc, max_neg_sc = eval_utils.compute_margin_loss(emb_scores, gd_ix, 0)
			mode_str = 'emb'
		else
			-- ensemble
			local scores = emb_scores + opt.lambda * lm_scores
			_, max_ix = torch.max(scores, 1)
			max_ix = max_ix[1]
			_, pos_sc, max_neg_sc = eval_utils.compute_margin_loss(scores, gd_ix, 0)
			mode_str = 'ensemble'
		end

		if pos_sc > max_neg_sc then accuracy = accuracy + 1 end
		loss_evals = loss_evals + 1

		-- add to predictions
		local gd_ann_id = img_ann_ids[gd_ix]
		local gd_box = loader.Anns[gd_ann_id]['box']
		local pred_ann_id = img_ann_ids[max_ix]
		local pred_box = loader.Anns[pred_ann_id]['box']
		local entry = {sent_id = sent_id, image_id = image_id, gd_ann_id = gd_ann_id, 
					   pred_ann_id = pred_ann_id, gd_box = gd_box, pred_box = pred_box}
		table.insert(predictions, entry)

		-- print
		local ix0 = data.bounds.it_pos_now - 1
		local ix1 = data.bounds.it_max
		print(string.format('%s-th: evaluating [%s] performance using [%s] ... image[%d/%d] sent[%d], acc=%.2f%%', 
			loss_evals, opt.split, mode_str, ix0, ix1, i, accuracy*100.0/loss_evals))
	end
	if data.bounds.wrapped then break end  -- we've used up images
end
print(string.format('accuracy = %.2f%%', accuracy/loss_evals*100))

-- save results
if not utils.file_exists('cache/box') then os.execute('mkdir cache/box') end
local cache_box_dataset_dir = path.join('cache/box', opt.dataset)
if not utils.file_exists(cache_box_dataset_dir) then os.execute('mkdir ' .. cache_box_dataset_dir) end
local cache_path = path.join(cache_box_dataset_dir, 'model_id' .. opt.id .. '_' .. opt.split .. '.json')
utils.write_json(cache_path, {predictions=predictions})












