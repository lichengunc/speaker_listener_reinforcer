require 'misc.LanguageModel'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

local eval_utils = {}
--[[

]]
function eval_utils.eval_loss(protos, loader, split, opt)

	local verbose = utils.getopt(opt, 'verbose', true)
	local lm_crit = nn.LanguageModelCriterion()
	if opt.gpuid >= 0 then lm_crit:cuda() end

	-- set evaluate mode
	for k, v in pairs(protos) do v:evaluate() end

	-- initialize
	loader:resetImageIterator(split)
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local accuracy = 0
	local predictions = {}

	-- evaluate 
	while true do

		-- fetch data for one image
		local data = loader:getImageBatch(split, opt)
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

			-- language generation loss (logprobs_i, sent_i)
			local gd_ix = gd_ixs[i]
			local lm_generation_loss = lm_crit:forward(logprobs[{ {}, {gd_ix}, {} }], seqz[{ {}, {i} }])

			-- language ranking margin loss 
			local lm_scores = -computeLosses(logprobs, sent_seqz)  -- (#img_ann_ids, )
			local lm_margin_loss, _, _  = eval_utils.compute_margin_loss(lm_scores, gd_ix, opt.lm_margin)

			-- embedding ranking margin loss
			local emb_margin_loss, pos_sc, max_neg_sc = eval_utils.compute_margin_loss(cossim, gd_ix, opt.emb_margin)

			-- combine the above
			loss_sum = loss_sum + (lm_generation_loss + lm_margin_loss + emb_margin_loss)
			loss_evals = loss_evals + 1
			if pos_sc > max_neg_sc then accuracy = accuracy + 1 end

			-- print
			if verbose then
				local ix0 = data.bounds.it_pos_now - 1
				local ix1 = data.bounds.it_max
				print(string.format('evaluating [%s] performance ... %d/%d sent[%d], acc=%.2f%%, (%.4f)',
					split, ix0, ix1, i, accuracy*100/loss_evals, loss_sum/loss_evals))
			end
		end
		if data.bounds.wrapped then break end  -- we've used up images
	end

	return loss_sum/loss_evals, accuracy/loss_evals
end
--[[
input:
- scores (n, 1)
- gd_ix
output:
- max(0, margin - max_neg_sc + pos_sc)
]]
function eval_utils.compute_margin_loss(scores, gd_ix, margin)
	local pos_sc = scores[gd_ix]
	scores[gd_ix] = -1e5
	local max_neg_sc = torch.max(scores)
	local loss = math.max(0, margin + max_neg_sc - pos_sc)
	return loss, pos_sc, max_neg_sc
end

function eval_utils.language_eval(predictions, split, opt)
	if not utils.file_exists('cache/lang') then os.execute('mkdir cache/lang') end
	local cache_lang_dataset_dir = path.join('cache/lang', opt.dataset)
	if not utils.file_exists(cache_lang_dataset_dir) then
		os.execute('mkdir '..cache_lang_dataset_dir)  
	end

	local file_name = opt.id .. '_' .. split .. '_beam' .. opt.beam_size
	local result_path = path.join('cache', 'lang', opt.dataset, file_name..'.json')
	utils.write_json(result_path, {predictions = predictions})
	-- call python to evaluate each sent with ground-truth sentences
	os.execute('python pyutils/python_eval_lang.py' .. 
		' --dataset ' .. opt.dataset ..
		' --model_id ' .. opt.id .. 
		' --beam_size ' .. opt.beam_size ..
		' --split ' .. split .. 
		' --write_result ' .. 1)
	-- return results
	local out_path = path.join(cache_lang_dataset_dir, file_name .. '_out.json')
	local out = utils.read_json(out_path)
	local result_struct = out['overall']
	return result_struct
end

return eval_utils







