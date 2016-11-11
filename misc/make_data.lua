local utils = require 'misc.utils'
require 'hdf5'
--[[
We will make up refs and sentences by generating expressions on some unlabeled anns.
We will save new data.json into cache/prepro+/dataset/ but leave data.h5 untouched.
because our dataloader is able to encode sent into seq by itself.
After saving data.json, we will reload DataLoader(new_data.json, data.h5).
Best do reshuffle each time loading new data.
]]
local make_data = {}

function make_data.make_new_data(protos, loader, split, num_to_label, opt)

	assert(split == 'train', 'We only make new data within training split.')
	
	-- set protos to evaluate mode, but DONT do :cuda()!!!
	for k, v in pairs(protos) do v:evaluate() end

	-- find new_sent_id and new_ref_id
	local max_sent_id, max_h5_id = 0, 0
	for _, sent in ipairs(loader.info.sentences) do 
		if sent['sent_id'] > max_sent_id then max_sent_id = sent['sent_id'] end
		if sent['h5_id'] > max_h5_id then max_h5_id = sent['h5_id'] end
	end
	local max_ref_id = 0
	for _, ref in ipairs(loader.info.refs) do 
		if ref['ref_id'] > max_ref_id then max_ref_id = ref['ref_id'] end
	end
	local new_sent_id = max_sent_id + 1
	local new_ref_id  = max_ref_id  + 1
	local new_h5_id   = max_h5_id + 1

	-- shuffle image_ids and reset iterator
	loader:shuffle_images(split)
	loader:resetImageIterator(split)

	-- make
	local N = 0
	while true do

		-- load image batch = {image_id, img_ann_ids, feats, sent_ids, gd_ixs, seqz, zseq}
		local data = loader:getImageBatch(split, opt)
		local image_id = data.image_id
		local img_ann_ids = data.img_ann_ids
		local feats = data.feats
		
		-- move to GPU
		if opt.gpuid >= 0 then for k = 1, #feats do feats[k] = feats[k]:cuda() end end
		
		-- prepare nr_ann_ids and nr_feats, "nr" means "non-ref"
		local nr_ann_ids = {}
		local nr_ixs = {}
		for ix, ann_id in ipairs(img_ann_ids) do
			if loader.annToRef[ann_id] == nil then
				table.insert(nr_ann_ids, ann_id)
				table.insert(nr_ixs, ix)
			end
		end
		if #nr_ann_ids == 0 then
			-- there is no non-ref ann_ids in this image, we jump to next image
			goto next_image
		end

		local nr_feats = {}
		for k, v in ipairs(feats) do
			nr_feats[k] = feats[k]:index(1, torch.LongTensor(nr_ixs))
		end

		-- forward nr_feats to sample beams
		local vis_enc_feats  = protos.vis_encoder:forward(nr_feats)
		local lang_enc_feats = vis_enc_feats:clone()  -- fake one
		local _, vis_emb_feats = unpack(protos.cca_embedding:forward{vis_enc_feats, lang_enc_feats})
		local vis_feats = protos.vis_combiner:forward{vis_enc_feats, vis_emb_feats} -- (n, 512+512)
		local sampleOpt = {sample_max = 1, beam_size = 3}
		local _, _, Done_beams = protos.lm:sample(vis_feats, sampleOpt)

		-- decode sents
		local nr_beam_sents = {}
		for k = 1, #nr_ann_ids do
			local beam_sents = {}
			for b, beam in ipairs(Done_beams[k]) do
				local beam_sent = loader:decode_sequence(beam['seq']:view(-1, 1))[1]
				table.insert(beam_sents, beam_sent)
			end
			table.insert(nr_beam_sents, beam_sents)
		end

		-- check (nr_ann_ids, nr_beam_sents)
		local vis_enc_feats = protos.vis_encoder:forward(feats)  -- (#img_ann_ids, 512)
		for k = 1, #nr_ann_ids do
			local flag = true
			local gd_ix = nr_ixs[k]
			local beam_sents = nr_beam_sents[k]
			local beam_zseq = loader:encode_sequence(beam_sents, {pad_zero = 'front'})  -- (seq_length, seq_per_ref) 
			local beam_seqz = loader:encode_sequence(beam_sents, {pad_zero = 'end'})
			if opt.gpuid >= 0 then beam_zseq = beam_zseq:cuda() end
			local beam_enc_feats = protos.lang_encoder:forward(beam_zseq)  -- (seq_per_ref, 512)
			-- check each beam_sent
			for b = 1, opt.seq_per_ref do
				-- if this is a wrong comprehension from embedding view
				local lang_enc_feats = beam_enc_feats[{ {b}, {} }]:expand(#img_ann_ids, opt.lang_encoding_size)  -- (#img_ann_ids, 512)
				local cossim, vis_emb_feats = unpack(protos.cca_embedding:forward{vis_enc_feats, lang_enc_feats})
				local _, max_ix = torch.max(cossim, 1)
				if max_ix[1] ~= gd_ix then 
					flag = false 
					break
				end
				-- if this is a wrong comprehension from language model view
				local seqz = beam_seqz[{{}, {b}}]:expand(loader.seq_length, #img_ann_ids)
				local vis_feats = protos.vis_combiner:forward{vis_enc_feats, vis_emb_feats}
				local logprobs = protos.lm:forward{vis_feats, seqz}:float()  -- (#img_ann_ids, )
				local lm_scores = -computeLosses(logprobs, seqz):float()  -- (#img_ann_ids, )
				local _, max_ix = torch.max(lm_scores, 1)
				if max_ix[1] ~= gd_ix then 
					flag = false 
					break
				end
				-- if this is a very long sentence
				local tokens = utils.split(beam_sents[b], ' ')
				if #tokens > loader.seq_length - 2 then  
					flag = false
					break
				end
			end
			-- if every beam_sent is correct, we add to data
			if flag == true then	
				-- add to sentences
				local ref_sent_ids = {}
				for b = 1, opt.seq_per_ref do
					local sent = {sent_id = new_sent_id,
								  sent = beam_sents[b],
								  tokens = utils.split(beam_sents[b], ' '),
								  h5_id = new_h5_id
								}
					table.insert(loader.info.sentences, sent)
					table.insert(ref_sent_ids, new_sent_id)
					new_sent_id = new_sent_id + 1
					new_h5_id   = new_h5_id   + 1
				end
				-- add to refs
				local ref = {ref_id = new_ref_id, 
							 ann_id = nr_ann_ids[k], 
							 box = loader.Anns[nr_ann_ids[k]]['box'],
							 image_id = image_id,
							 split = 'train',
							 category_id = loader.Anns[nr_ann_ids[k]]['category_id'],
							 sent_ids = ref_sent_ids
							}
				table.insert(loader.info.refs, ref)
				new_ref_id = new_ref_id + 1

				-- print
				N = N + 1
				local ix0 = data.bounds.it_pos_now - 1
				local ix1 = data.bounds.it_max
				print(string.format('%s/%s sampled for ann_id[%s][%s], image[%d/%d], its sentences are ...', 
					N, num_to_label, nr_ann_ids[k], loader.ix_to_cat[tostring(loader.Anns[nr_ann_ids[k]]['category_id'])],
					ix0, ix1))
				print(beam_sents)
				if num_to_label > 0 and N >= num_to_label then break end
			else
				-- print(string.format('ann_id[%s] discared...', nr_ann_ids[k]))
			end
		end		

		-- check if wrapped out
		if data.bounds.wrapped then break end
		if num_to_label > 0 and N >= num_to_label then break end

		-- for immergency we jump to next image
		::next_image::
	end

	-- save data.json
	if not utils.file_exists('cache/prepro+') then os.execute('mkdir cache/prepro+') end
	if not utils.file_exists('cache/prepro+/' .. opt.dataset) then 
		os.execute('mkdir cache/prepro+/' .. opt.dataset)
	end
	local json_path = path.join('cache/prepro+', opt.dataset, 'data.json')
	utils.write_json(json_path, loader.info)
	print(string.format('%s new refs collected.', N))
	print(string.format('new data.json saved in %s', json_path))

	-- save data.h5
	

end

return make_data





