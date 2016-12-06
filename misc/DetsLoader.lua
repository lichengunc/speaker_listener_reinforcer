require 'hdf5'
local utils = require 'misc.utils'

-- Read json file has
-- 0. refs: list of {ref_id, ann_id, box, image_id, split, category_id, sent_ids}
-- 1. images: list of {image_id, ref_ids, ann_ids, file_name, width, height, h5_id}
-- 2. anns: list of {ann_id, category_id, image_id, box, h5_id}
-- 3. sentences: list of {sent_id, tokens, sent, h5_id}
-- 4. ix_to_word
-- 5. word_to_ix
-- 6. ix_to_cat

local DetsLoader = torch.class('DetsLoader')

function DetsLoader:__init(opt)

	-- load the json file which contains info about the dataset
	print('DetsLoader loading data.json: ', opt.data_json)
	self.info = utils.read_json(opt.data_json)
	self.ix_to_word = self.info.ix_to_word
	self.word_to_ix = self.info.word_to_ix
	self.vocab_size = utils.count_keys(self.ix_to_word)
	print('vocab size is ' .. self.vocab_size)
	self.ix_to_cat = self.info.ix_to_cat
	print('object category size is ' .. utils.count_keys(self.ix_to_cat))
	self.images = self.info.images
	self.anns = self.info.anns
	self.refs = self.info.refs
	print('We have ' .. #self.images .. ' images.')
	print('We have ' .. #self.anns .. ' anns.')
	print('We have ' .. #self.refs .. ' refs.')

	-- open hdf5 file
	print('DetsLoader loading data.h5: ', opt.data_h5)
	self.data_h5 = hdf5.open(opt.data_h5, 'r')
	local seq_size = self.data_h5:read('/seqz_labels'):dataspaceSize()
	self.seq_length = seq_size[2]
	print('max sequence length in data is ' .. self.seq_length)

	-- construct Refs, Images, Anns, Sentences and annToRef
	local Refs, Images, Anns, Sentences, annToRef = {}, {}, {}, {}, {}
	for i, ref in ipairs(self.info.refs) do Refs[ref['ref_id']] = ref; annToRef[ref['ann_id']] = ref end
	for i, image in ipairs(self.info.images) do Images[image['image_id']] = image end
	for i, ann in ipairs(self.info.anns) do Anns[ann['ann_id']] = ann end
	for i, sent in ipairs(self.info.sentences) do Sentences[sent['sent_id']] = sent end
	self.Refs, self.Images, self.Anns, self.Sentences, self.annToRef = Refs, Images, Anns, Sentences, annToRef

	-- construct sentToRef
	self.sentToRef = {}
	for i = 1, #self.refs do
		local ref = self.refs[i]
		for _, sent_id in ipairs(ref['sent_ids']) do self.sentToRef[sent_id] = ref end
	end

	-- loading dets = [{det_id, box, image_id, category_id, h5_id}]
	-- make Dets, and add det_ids to each image
	print('DetsLoader loading dets.json: ', opt.dets_json)
	self.dets = utils.read_json(opt.dets_json)['dets']
	local Dets = {}
	for i, det in ipairs(self.dets) do
		local det_id = det['det_id']
		local image_id = det['image_id']
		Dets[det_id] = det
		if Images[image_id].det_ids ~= nil then
			table.insert(Images[image_id].det_ids, det_id)
		else
			Images[image_id].det_ids = {det_id}
		end
	end
	self.Dets = Dets
	print('We have '..#self.dets..' detected boxes.')

	-- ref iterators for each split
	self.split_ix = {}
	self.iterators = {}
	for ref_id, ref in pairs(self.Refs) do
		local split = ref['split']
		if not self.split_ix[split] then
			self.split_ix[split] = {}
			self.iterators[split] = 1
		end
		table.insert(self.split_ix[split], ref_id)
	end
	for k, v in pairs(self.split_ix) do
		print(string.format('assigned %d refs to split %s', #v, k))
	end
	-- sent iterators for each split
	self.sent_split_ix = {}
	self.sent_iterators = {}
	for sent_id, sent in pairs(self.Sentences) do
		local split = self.sentToRef[sent_id]['split']
		if not self.sent_split_ix[split] then
			self.sent_split_ix[split] = {}
			self.sent_iterators[split] = 1
		end
		table.insert(self.sent_split_ix[split], sent_id)
	end
	for k, v in pairs(self.sent_split_ix) do
		print(string.format('assigned %d sents to split %s.', #v, k))
	end
	-- image iterators for each split
	self.img_split_ix = {}
	self.img_iterators = {}
	for image_id, image in pairs(self.Images) do
		-- collect split_names appeared in this image
		local split_names = {}
		for _, ref_id in ipairs(image['ref_ids']) do
			split_names[#split_names+1] = self.Refs[ref_id]['split']
		end
		split_names = utils.unique(split_names)
		-- add to iterators
		for _, split in ipairs(split_names) do
			if not self.img_split_ix[split] then
				self.img_split_ix[split] = {}
				self.img_iterators[split] = 1
			end
			table.insert(self.img_split_ix[split], image_id)
		end
	end
	for k, v in pairs(self.img_split_ix) do
		print(string.format('assigned %d images to split %s', #v, k))
	end
end
-- load pre-computed graphs = [{image_id, ann_ids, cossim}]
function DetsLoader:load_graph(graph_path)
	local graphs = utils.read_json(graph_path)
	local Graphs = {}
	for _, graph in ipairs(graphs) do
		Graphs[graph['image_id']] = graph
	end
	self.graphs = graphs
	self.Graphs = Graphs
	print(string.format('graph [%s] being used...', graph_path))
end
-- load different kinds of feats.h5
function DetsLoader:loadFeats(featsOpt)
	-- register to self.feats
	self.feats = {}
	for key, feats_h5 in pairs(featsOpt) do
		if utils.file_exists(feats_h5) then
			print('FeatLoader loading ' .. feats_h5)
			self.feats[key] = hdf5.open(feats_h5, 'r')
		end
	end
end
-- shuffle ann_ids
function DetsLoader:shuffle(split)
	for i = #self.split_ix[split], 2, -1 do
		j = math.random(i)
		self.split_ix[split][i], self.split_ix[split][j] = self.split_ix[split][j], self.split_ix[split][i]
	end
end
-- shuffle image_ids
function DetsLoader:shuffle_images(split)
	for i = #self.img_split_ix[split], 2, -1 do
		j = math.random(i)
		self.img_split_ix[split][i], self.img_split_ix[split][j] = 
			self.img_split_ix[split][j], self.img_split_ix[split][i]
	end
end
-- reset iterator [split] to start
function DetsLoader:resetIterator(split)
	self.iterators[split] = 1
end
-- reset sent iterator [split] to start
function DetsLoader:resetSentIterator(split)
	self.sent_iterators[split] = 1
end
-- reset image iterator [split] to start
function DetsLoader:resetImageIterator(split)
	self.img_iterators[split] = 1
end
-- vocab size
function DetsLoader:getVocabSize()
	return self.vocab_size
end
-- get ix_to_word
function DetsLoader:getVocab()
	return self.ix_to_word
end
-- get seq_length
function DetsLoader:getSeqLength()
	return self.seq_length
end

-- fetch feats = {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}
function DetsLoader:fetch_det_feats(batch_det_ids, expand_size, opt)

	local cxt_feats = torch.FloatTensor(#batch_det_ids*expand_size, 4096)
	local ann_feats = torch.FloatTensor(#batch_det_ids*expand_size, 4096)
	local lfeats = torch.FloatTensor(#batch_det_ids*expand_size, 5)
	local dif_ann_feats = torch.FloatTensor(#batch_det_ids*expand_size, 4096)
	local dif_lfeats = torch.FloatTensor(#batch_det_ids*expand_size, 5*opt.dif_num)

	for i, det_id in ipairs(batch_det_ids) do
		-- start
		local il = (i-1)*expand_size + 1
		-- fetch feat
		local cf, af, lf = self:fetch_det_feat(det_id, opt)
		cxt_feats[{ {il, il+expand_size-1} }] = cf:expand(expand_size, 4096)
		ann_feats[{ {il, il+expand_size-1} }] = af:expand(expand_size, 4096)
		lfeats[{ {il, il+expand_size-1} }] = lf:view(1, -1):expand(expand_size, 5)
		-- fetch dif_feat
		local df, dlf = self:fetch_det_dif_feat(det_id, opt)
		dif_ann_feats[{ {il, il+expand_size-1} }] = df:view(1, -1):expand(expand_size, 4096)
		dif_lfeats[{ {il, il+expand_size-1} }] = dlf:view(1, -1):expand(expand_size, 5*opt.dif_num)
	end

	return {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}
end

-- fetch both seqz and zseq, be careful about the order !!!
function DetsLoader:fetch_seqs(batch_sent_ids, opt)

	assert(opt.pad_zero == 'front' or opt.pad_zero == 'end')
	local seq = torch.LongTensor(self.seq_length, #batch_sent_ids)
	
	for i, sent_id in ipairs(batch_sent_ids) do

		local sent = self.Sentences[sent_id]
		if sent['h5_id'] ~= nil then
			-- we read from data.h5
			local dname
			if opt.pad_zero == 'front' then dname = '/zseq_labels' else dname = '/seqz_labels' end
			seq[{ {}, {i} }] = self.data_h5:read(dname):partial({sent['h5_id'], sent['h5_id']}, {1, self.seq_length})
		else
			-- we encode sent string
			seq[{ {}, {i} }] = self:encode_sequence({sent['sent']}, opt)
		end
	end
	return seq
end
-- fetch (num_sents) sent_ids given ref_id
function DetsLoader:fetch_sent_ids(ref_id, num_sents)
	
	local ref = self.Refs[ref_id]
	-- pick sent_ids from ref['sent_ids']
	local picked_sent_ids = {}
	if #ref['sent_ids'] < num_sents then
		picked_sent_ids = utils.copyTable(ref['sent_ids'])
		for q = 1, num_sents - #ref['sent_ids'] do
			local ix = torch.random(1, #ref['sent_ids'])
			local sent_id = ref['sent_ids'][ix]
			table.insert(picked_sent_ids, sent_id)
		end
	else
		ref_sent_ids = utils.shuffleTable(utils.copyTable(ref['sent_ids']))
		for q = 1, num_sents do
			local sent_id = ref_sent_ids[q]
			table.insert(picked_sent_ids, sent_id)
		end
	end
	-- return
	return picked_sent_ids
end
--[[
For given ref_ann_id, we return
- st_ann_ids: same-type neighbouring ann_ids (not including itself)
- dt_ann_ids: different-type neighbouring ann_ids
Ordered by distance to the input ann_id
]]
function DetsLoader:fetch_neighbour_ids(ref_ann_id)
	local ref_ann = self.Anns[ref_ann_id]
	local x, y, w, h = unpack(ref_ann['box'])
	local rx, ry = x+w/2, y+h/2
	local function compare(ann_id1, ann_id2) 
		local x, y, w, h = unpack(self.Anns[ann_id1]['box'])
		local ax1, ay1 = x+w/2, y+h/2
		local x, y, w, h = unpack(self.Anns[ann_id2]['box'])
		local ax2, ay2 = x+w/2, y+h/2
		return (rx-ax1)^2 + (ry-ay1)^2 < (rx-ax2)^2 + (ry-ay2)^2  -- closer --> former
	end
	local image = self.Images[ref_ann['image_id']]
	local ann_ids = utils.copyTable(image['ann_ids'])
	table.sort(ann_ids, compare)

	local st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = {}, {}, {}, {}
	for i = 1, #ann_ids do
		local ann_id = ann_ids[i]
		if ann_id ~= ref_ann_id then
			if self.Anns[ann_id]['category_id'] == ref_ann['category_id'] then
				table.insert(st_ann_ids, ann_id)
				if self.annToRef[ann_id] ~= nil then table.insert(st_ref_ids, self.annToRef[ann_id]['ref_id']) end 
			else
				table.insert(dt_ann_ids, ann_id)
				if self.annToRef[ann_id] ~= nil then table.insert(dt_ref_ids, self.annToRef[ann_id]['ref_id']) end
			end
		end
	end
	return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids
end
-- return cxt_feat, ann_feat, lfeat
function DetsLoader:fetch_det_feat(det_id, opt)
	local use_context = utils.getopt(opt, 'use_context', 1) -- 0.none, 1.img, 2.window2, 3.window3
	local use_ann = utils.getopt(opt, 'use_ann', 1) 				-- 1.vgg, 2.att, 3.frc
	local det = self.Dets[det_id]
	local image = self.Images[det['image_id']]	
	-- fetch cxt_feat
	local cxt_feat = torch.FloatTensor(1, 4096)
	if use_context == 1 then cxt_feat = self.feats['img']:read('/img_feats'):partial(image['h5_id'], {1, 4096}) end
	if use_context ~= 1 then error('Not implemented') end
	-- fetch ann_feat
	local ann_feat = torch.FloatTensor(1, 4096)
	if use_ann == 1 then ann_feat = self.feats['det']:read('/det_feats'):partial(det['h5_id'], {1, 4096}) end
	if use_ann ~= 1 then error('Not implemented') end
	-- compute lfeats
	local x, y, w, h = unpack(det['box'])
	local iw, ih = image['width'], image['height']
	local lfeat = torch.FloatTensor{x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)}
	-- return
	return cxt_feat, ann_feat, lfeat
end
--[[
if there is same-category det_id, we aggregate its difference feature.
if not, we encode with zeros.
For relative location feature, we encode the whole image with binary mask, and take out region centered by object
or we encode relative location of the nearest 5 objects using 25-D vector. Zeros are padded if any.
]]
function DetsLoader:fetch_det_dif_feat(det_id, opt)
	local dif_ann = utils.getopt(opt, 'dif_ann')         -- 1.vgg, 2.att, 3.frc
	local dif_pool = utils.getopt(opt, 'dif_pool')			 -- 0.none, 1.mean, 2.max, 3.min, 4.weighted
	local dif_source = utils.getopt(opt, 'dif_source')   -- 1.st_anns, 2.dt_anns, 3.st_anns+dt_anns
	local dif_num = utils.getopt(opt, 'dif_num')         -- number of nearby objects to be considerred
	-- compute feat
	local dif_det_feat = torch.FloatTensor(4096):zero()
	local dif_lfeat = torch.FloatTensor(dif_num*5):zero()
	local st_det_ids, dt_det_ids = self:fetch_neighbour_det_ids(det_id)
	local cand_det_ids
	if dif_source == 1 then cand_det_ids = st_det_ids end
	if dif_source == 2 then cand_det_ids = dt_det_ids end
	if dif_source == 3 then cand_det_ids = utils.combineTable(st_det_ids, dt_det_ids) end
	if #cand_det_ids ~= 0 then
		-- get cand_det_feats
		if #cand_det_ids > dif_num then for k=dif_num+1, #cand_det_ids do table.remove(cand_det_ids, dif_num+1) end end
		local cand_det_feats = torch.FloatTensor(#cand_det_ids, 4096)
		for j, cand_det_id in ipairs(cand_det_ids) do
			cand_det_feats[j] = self.feats['det']:read('/det_feats'):partial(self.Dets[cand_det_id]['h5_id'], {1, 4096})
		end
		-- get ref_det_feat
		local ref_det_feat = self.feats['det']:read('/det_feats'):partial(self.Dets[det_id]['h5_id'], {1, 4096})
		-- compute pooled feat
		cand_det_feats = cand_det_feats - ref_det_feat:expandAs(cand_det_feats)
		dif_det_feat = torch.mean(cand_det_feats, 1)
		-- compute dif_lfeat
		local image = self.Images[self.Dets[det_id]['image_id']]
		local iw, ih = image['width'], image['height']
		local rbox = self.Dets[det_id]['box']
		local rcx, rcy, rw, rh = rbox[1]+rbox[3]/2, rbox[2]+rbox[4]/2, rbox[3], rbox[4]
		for j = 1, math.min(5, #cand_det_ids) do
			local cbox = self.Dets[cand_det_ids[j]]['box']
			local cx1, cy1, cw, ch = cbox[1], cbox[2], cbox[3], cbox[4]
			dif_lfeat[{ {(j-1)*5+1, j*5} }] = torch.FloatTensor{ (cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh) } -- we don't bother normalizing here.
		end
	end
	return dif_det_feat, dif_lfeat
end
--[[
For given det_id, we return
- st_ann_ids: same-type neighbouring ann_ids (not including itself)
- dt_ann_ids: different-type neighbouring ann_ids
Ordered by distance to the input ann_id
]]
function DetsLoader:fetch_neighbour_det_ids(ref_det_id)
	local ref_det = self.Dets[ref_det_id]
	local x, y, w, h = unpack(ref_det['box'])
	local rx, ry = x+w/2, y+h/2
	local function compare(det_id1, det_id2) 
		local x, y, w, h = unpack(self.Dets[det_id1]['box'])
		local ax1, ay1 = x+w/2, y+h/2
		local x, y, w, h = unpack(self.Dets[det_id2]['box'])
		local ax2, ay2 = x+w/2, y+h/2
		return (rx-ax1)^2 + (ry-ay1)^2 < (rx-ax2)^2 + (ry-ay2)^2  -- closer --> former
	end
	local image = self.Images[ref_det['image_id']]
	local det_ids = utils.copyTable(image['det_ids'])
	table.sort(det_ids, compare)

	local st_det_ids, dt_det_ids = {}, {}
	for i = 1, #det_ids do
		local det_id = det_ids[i]
		if det_id ~= ref_det_id then
			if self.Dets[det_id]['category_id'] == ref_det['category_id'] then
				table.insert(st_det_ids, det_id)
			else
				table.insert(dt_det_ids, det_id)
			end
		end
	end
	return st_det_ids, dt_det_ids
end
--[[
input:  {n string of sent}
output: long (seq_length, n), zero padded in the front
        or long (seq_length, n), zero padded in the end
]]
function DetsLoader:encode_sequence(sents, opt)
	local pad_zero = utils.getopt(opt, 'pad_zero')
	assert(pad_zero == 'front' or pad_zero == 'end')

	if pad_zero == 'front' then
		local seqs = {}
		for n, sent in ipairs(sents) do
			local tokens = utils.split(sent, ' ')
			local seq = torch.LongTensor(self.seq_length):zero()
			local start_ix = math.max(self.seq_length-#tokens, 0)
			for i, token in ipairs(tokens) do
				if i <= self.seq_length then
					seq[start_ix+i] = self.word_to_ix[token]
				end 
			end
			table.insert(seqs, seq)
		end
		seqs = torch.cat(seqs, 2):long()
		return seqs
	else
		local seqs = {}
		for n, sent in ipairs(sents) do
			local tokens = utils.split(sent, ' ')
			local seq = torch.LongTensor(self.seq_length):zero()
			for i = 1, math.min(#tokens, self.seq_length) do
				seq[i] = self.word_to_ix[tokens[i]]
			end
			table.insert(seqs, seq)
		end
		seqs = torch.cat(seqs, 2):long()
		return seqs
	end
end
--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function DetsLoader:decode_sequence(seq)
	local D,N = seq:size(1), seq:size(2)
	local out = {}
	for i=1,N do
		local txt = ''
		local flag = false
		for j=1,D do
			local ix = seq[{j,i}]
			if ix ~= 0 then
				local word = self.ix_to_word[tostring(ix)]
				if not word then break end -- END token, likely. Or null token
				if flag == true then txt = txt .. ' ' end
				txt = txt .. word
				flag = true
			end
		end
		table.insert(out, txt)
	end
	return out
end
-- convert seqz to zseq
function DetsLoader:seqz_to_zseq(seqz)
	local dtype = seqz:type()
	local sents = self:decode_sequence(seqz)
	local zseq  = self:encode_sequence(sents, {pad_zero = 'front'})
	zseq = zseq:type(dtype)
	return zseq
end
--[[
Get img_ann_ids given next image_id
We can check the margin of gd_ann_id over neg_ann_ids for each sent.
return a image batch:
- img_det_ids : n det_ids in this image
- feats       : table of (n, dim) feats
- sent_ids    : m sent_ids 
- gd_boxes    : m gd_box corresponding to each sent_id
- seqz        : (seq_length, m)
- zseq        : (seq_length, m)
]]
function DetsLoader:getImageBatch(split, opt)
	-- feat option
	local use_context = utils.getopt(opt, 'use_context')
	local use_ann = utils.getopt(opt, 'use_ann')
	-- dif feat option
	local dif_ann = utils.getopt(opt, 'dif_ann')  	-- 0.none, 1.vgg, 2.att, 3.frc
	local dif_pool = utils.getopt(opt, 'dif_pool')  -- 1.mean, 2.max, 3.min
	local dif_source = utils.getopt(opt, 'dif_source') -- 1.st_anns, 2.dt_anns, 3.st_anns+dt_anns
	local dif_num = utils.getopt(opt, 'dif_num')  -- number of nearby objects to be considerred
	-- move img_split_ix
	local wrapped = false
	local img_split_ix = self.img_split_ix[split]
	local mi = self.img_iterators[split]
	local image_id = img_split_ix[mi]
	local mi_next = mi + 1
	if mi_next > #img_split_ix then wrapped = true end
	self.img_iterators[split] = mi_next

	-- current image
	local image = self.Images[image_id]
	local det_ids = image['det_ids']
	if det_ids == nil then det_ids = {1} end -- fake det_ids in case nothing was detected in this image
	local feats = self:fetch_det_feats(det_ids, 1, opt)

	-- sent_ids and gd_ref_boxs
	local sent_ids = {}
	local gd_ref_boxs = {}
	local img_ref_ids = image['ref_ids'] 
	for ix, ref_id in ipairs(img_ref_ids) do
		local ref = self.Refs[ref_id]
		if ref['split'] == split then
			for _, sent_id in ipairs(ref['sent_ids']) do
				table.insert(sent_ids, sent_id)
				table.insert(gd_ref_boxs, ref['box'])
			end
		end
	end

	-- fetch seqz and zseq for each sent_id
	local seqz = self:fetch_seqs(sent_ids, {pad_zero = 'end'})
	local zseq = self:fetch_seqs(sent_ids, {pad_zero = 'front'})

	-- return
	local data = {}
	data.image_id = image_id
	data.img_det_ids = det_ids   
	data.feats = feats  
	data.sent_ids = sent_ids
	data.gd_boxs = gd_ref_boxs
	data.seqz = seqz
	data.zseq = zseq 
	data.bounds = {it_pos_now = self.img_iterators[split], it_max = #img_split_ix, wrapped = wrapped}
	return data
end




























