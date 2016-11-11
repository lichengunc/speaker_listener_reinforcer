require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.modules.SplitEmbedding'
require 'misc.modules.SplitGeneration'
require 'misc.modules.FeatExpander'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local eval_utils = require 'misc.eval_utils'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate Referring Expression Generation')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset+splitBy')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-id', '', 'model id to be evaluated')  -- corresponding to opt.id in train.lua
-- Basic options
cmd:option('-batch_size', 32, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
-- For evaluation on refer dataset for some split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- misc
cmd:option('-verbose', 1, 'print during evaluation?')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-write_html', 0, '1 = write html for visualization, saved in cache/vis/lang/dataset_splitBy')
cmd:text()

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
			   'dif_source', 'dif_pool', 'seq_per_ref'}
for k, v in pairs(fetch) do opt[v] = checkpoint.opt[v] end
-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
-- set mode
for k, v in pairs(protos) do
	print('protos has ' .. k)  -- vis_encoder, lang_encoder, cca_embedding, vis_combiner, lm, we, split_emb, split_lm
	v:evaluate()  -- set evalute mode
end

-- add feat expander and lm criterion
protos.expander = nn.FeatExpander(opt.seq_per_ref)
protos.lm_crit  = nn.LanguageModelCriterion()

-- ship to GPU
if opt.gpuid >= 0 then 
	for k, v in pairs(protos) do v:cuda() end
end

-- initialize
loader:resetIterator(opt.split)
local n = 0
local loss_sum = 0
local loss_evals = 0
local predictions = {}
local Ref_ids = {}  -- record if ref_id was already taken

-- evaluate 
while true do

	local data = loader:getTestBatch(opt.split, opt)
	local ref_ids = data.ref_ids
	local feats = data.feats  -- {(n, dim), (), ...}
	local seqz  = data.seqz   -- (seq_length, n * seq_per_ref)
	local zseq  = data.zseq
	n = n + opt.batch_size

	-- ship to GPU
	if opt.gpuid >= 0 then 
		for k = 1, #feats do feats[k] = feats[k]:cuda() end
		zseq = zseq:cuda()
	end

	-- forward by expanding vis_enc_feats seq_per_ref times
	local vis_enc_feats  = protos.vis_encoder:forward(feats)  -- (n, 512)
	local expanded_vis_enc_feats = protos.expander:forward(vis_enc_feats)  -- (n * seq_per_ref, 512+512)
	local lang_enc_feats = protos.lang_encoder:forward(zseq)
	local cossim, vis_emb_feats = unpack(protos.cca_embedding:forward{expanded_vis_enc_feats, lang_enc_feats})
	local vis_feats = protos.vis_combiner:forward{expanded_vis_enc_feats, vis_emb_feats}  -- (n * seq_per_ref, 512+512)
	local logprobs  = protos.lm:forward{vis_feats, seqz} -- (seq_length+1, n * seq_per_ref, Mp1)
	local loss = protos.lm_crit:forward(logprobs, seqz)
	loss_sum = loss_sum + loss
	loss_evals = loss_evals + 1

	-- forward the model to also get samples for each image
	local vis_enc_feats  = protos.vis_encoder:forward(feats)  -- (n, 512)
	local lang_enc_feats = vis_enc_feats:clone() -- fake one
	local _, vis_emb_feats = unpack(protos.cca_embedding:forward{vis_enc_feats, lang_enc_feats})
	local vis_feats = protos.vis_combiner:forward{vis_enc_feats, vis_emb_feats}  -- (n, 512+512)
	-- sample
	local sampleOpt = {sample_max = opt.sample_max, beam_size = opt.beam_size}
	local sampled_seq, _, Done_beams = protos.lm:sample(vis_feats, sampleOpt) 
	local sents = loader:decode_sequence(sampled_seq)
	for k = 1, #sents do

		local ref_id = ref_ids[k]
		local sent = sents[k]
		if Ref_ids[ref_id] == nil then  
			Ref_ids[ref_id] = 'done'  -- we don't do twice for the same ref_id
			local entry = {ref_id = ref_id, sent = sent}
			if opt.verbose then print(string.format('ref_id%s: %s', entry.ref_id, entry.sent)) end
			if opt.beam_size > 1 then 
				-- add beams to entry if beam_size > 1
				local beams = {}
				for b, beam in ipairs(Done_beams[k]) do
					-- beam contains {seq, logps, ppl, p}
					local beam_sent = loader:decode_sequence(beam['seq']:view(-1, 1))[1]
					beams[b] = {sent=beam_sent, ppl=beam['ppl'], logp=beam['p']}
					if opt.verbose then
						print(string.format('beam[%s]: %s (logp=%.2f, ppl=%.2f)', b, beam_sent, beam['p'], beam['ppl']))
					end
				end
				entry['beams'] = beams
			end
			table.insert(predictions, entry)
		end
	end
	-- if we wrapped around the split or used up num_images than bail
	local ix0 = data.bounds.it_pos_now
	local ix1 = math.min(data.bounds.it_max, opt.num_images)
	if opt.verbose then
		print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
	end

	if data.bounds.wrapped then break end -- the split ran out of data, lets break out
	if opt.num_images >= 0 and n >= opt.num_images then break end -- we've used enough images
	if loss_evals % 200 == 0 then collectgarbage() end

end

print('loss: ', loss_sum / loss_evals)
local lang_stats
if opt.language_eval == 1 then
	lang_stats = eval_utils.language_eval(predictions, opt.split, opt)
	print(lang_stats)
end

-- write html
if opt.write_html == 1 then
	os.execute('python scripts/visualize_lang.py ' .. '--dataset_splitBy ' .. opt.dataset .. ' --model_id ' .. opt.id .. ' --split ' .. opt.split) 
end



































