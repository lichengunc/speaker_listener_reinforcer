--[[
Before running this script, make sure to run "eval_lang.lua -beam_size 10" which saves
predictions in cache/lang/dataset/model_id_val_beam10.json, which stores
{'predictions': [ref_id, sent, beams]}, where the beams = [{ppl, logp, sent}]

Within each image, we compute score (beam_size, #ref_ids) for each ref_id, 
score_ij indicates sc(beam_i, ref_j).
]]
require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.modules.SplitEmbedding'
require 'misc.modules.SplitGeneration'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Rank Referring Expression Generation')
cmd:text()
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset+splitBy')
cmd:option('-id', '0', 'model id to be evaluated')
cmd:option('-beam_size', 10, 'beam_size being evaluated')
cmd:option('-split', 'val', 'what split to cross validate')
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
-- Load generated sentences
-------------------------------------------------------------------------------
local beam_json = path.join('cache/lang', opt.dataset, opt.id .. '_' .. opt.split .. '_beam' .. opt.beam_size .. '.json')
local beam_data = utils.read_json(beam_json)
local beam_data = beam_data['predictions']
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

-- get visemb and langemb
local sub_embedding = net_utils.extract_sub_embedding(protos)
local visemb  = sub_embedding.visemb
local langemb = sub_embedding.langemb

-- ship to GPU and set to evaluate mode
if opt.gpuid >= 0 then
	visemb:cuda();  visemb:evaluate()
	langemb:cuda(); langemb:evaluate()
end
-------------------------------------------------------------------------------
-- Compute score for each ref_id
-------------------------------------------------------------------------------
local ref_to_beams = {}   -- {ref_id: beams}
local img_to_ref_ids = {} -- {image_id: ref_ids}

for _, item in ipairs(beam_data) do

	-- each item = {ref_id, sent, beams}, where beams = [{ppl, logp, sent}]
	local ref_id = item['ref_id']
	ref_to_beams[ref_id] = item['beams']

	-- add to img_to_ref_ids
	local image_id = loader.Refs[ref_id]['image_id']
	if img_to_ref_ids[image_id] == nil then img_to_ref_ids[image_id] = {} end
	table.insert(img_to_ref_ids[image_id], ref_id)
end

-- compute score (beam_size, #img_ref_ids) for each ref_id
local img_to_ref_confusion = {}

for image_id, img_ref_ids in pairs(img_to_ref_ids) do

	img_to_ref_confusion[image_id] = {}

	-- fetch feats for img_ref_ids (in this image)
	local img_ann_ids = {}
	for _, ref_id in ipairs(img_ref_ids) do
		table.insert(img_ann_ids, loader.Refs[ref_id]['ann_id'])
	end
	local feats = loader:fetch_feats(img_ann_ids, 1, opt)

	-- compute score for each ref_id
	for _, ref_id in ipairs(img_ref_ids) do
		
		-- fetch zseq for the 10 beam sents
		local beams = ref_to_beams[ref_id]
		local sents = {}
		for _, beam in ipairs(beams) do
			table.insert(sents, beam['sent'])
		end
		local zseq = loader:encode_sequence(sents, {pad_zero='front'})  -- (seq_length, beam_size)

		-- ship to GPU
		if opt.gpuid >= 0 then
			for k = 1, #feats do feats[k] = feats[k]:cuda() end
			zseq = zseq:cuda()
		end

		-- compute score (beam_size, #img_ref_ids)
		local vis_emb_feats  = visemb:forward(feats):float()  -- (#img_ref_ids, d)
		local lang_emb_feats = langemb:forward(zseq):float()  -- (beam_size, d)
		local score = nn.MM(false, true):float():forward{lang_emb_feats, vis_emb_feats}
		score = utils.tensor_to_table(score)  -- (beam_size, #img_ref_ids)

		-- add to img_to_ref_confusion
		img_to_ref_confusion[image_id][tostring(ref_id)] = score  -- use tostring to avoid ascii issue, don't know why...
	end
end

-- save confusion scores
local output_json = path.join('cache/lang', opt.dataset, opt.id .. '_' .. opt.split .. '_beam' .. opt.beam_size .. '_confusion.json')
utils.write_json(output_json, {img_to_ref_confusion = img_to_ref_confusion, 
							   img_to_ref_ids = img_to_ref_ids})













