require 'image'
require 'nn'
require 'rnn'
require 'nngraph'
local utils = require 'misc.utils'
local net_utils = {}

-- take a raw CNN from caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
	local layer_num = utils.getopt(opt, 'layer_num', 38)
	-- copy over the first layer_num layer of the CNN
	local cnn_part = nn.Sequential()
	for i = 1, layer_num do
		local layer = cnn:get(i)
		if i == 1 then
			local w = layer.weight:clone()
			layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
			layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
		end
		cnn_part:add(layer)
	end
	-- return 
	return cnn_part
end
-- preprocessing of one raw image (1-255) of shape (3, H, W)
-- resize to (3, 224, 224) and subtract mean
function net_utils.prepro_img(raw_img, on_gpu)
	local h, w = raw_img:size(2), raw_img:size(3)
	local cnn_input_size = 224
	-- tile image to 3 channels
	if raw_img:size(1) == 1 then
		raw_img = raw_img:expand(3, h, w)
	end
	-- resize to (3, 224, 224)
	local img
	if h ~= cnn_input_size or w ~= cnn_input_size then
		img = image.scale(raw_img, cnn_input_size, cnn_input_size)
	else
		img = raw_img
	end
	-- ship to gpu
	if on_gpu then img = img:cuda() else img = img:float() end
	-- subtract vgg mean
	local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(3,1,1) -- in RGB order
	vgg_mean = vgg_mean:typeAs(img)
	img:add(-1, vgg_mean:expandAs(img))

	return img
end
-- preprocessing on raw ann (1-255) shape (3, H, W)
-- maintain scale and pad with zeros fitting (3, 224, 224)
function net_utils.prepro_ann(raw_ann, on_gpu)
	local h, w = raw_ann:size(2), raw_ann:size(3)
	local cnn_input_size = 224
	-- tile image to 3 channels
	if raw_ann:size(1) == 1 then
		raw_ann = raw_ann:expand(3, h, w)
	end
	-- resize to 224
	local ann_img, nh, nw
	if h <= w then
    	nh, nw = math.floor(math.max(1, math.min(cnn_input_size/w*h, cnn_input_size))), cnn_input_size
    	ann_img = image.scale(raw_ann, nw, nh)
	else
		nh, nw = cnn_input_size, math.floor(math.max(1, math.min(cnn_input_size/h*w, cnn_input_size)))
		ann_img = image.scale(raw_ann, nw, nh)
	end
	-- ship to gpu
	if on_gpu then ann_img = ann_img:cuda() else ann_img = ann_img:float() end

	-- subtract vgg mean
	local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(3,1,1) -- in RGB order
	vgg_mean = vgg_mean:typeAs(ann_img)
	ann_img:add(-1, vgg_mean:expandAs(ann_img))

	-- feed in to (3, 224, 224) zeros
	local pad_img = torch.zeros(3, 224, 224):typeAs(ann_img)
	local x1, x2
	if nh <= nw then
		x1 = math.max(1, math.floor((cnn_input_size-nh)/2))
		x2 = x1 + nh - 1 
		pad_img[{ {}, {x1, x2}, {} }] = ann_img
	else 
		y1 = math.max(1, math.floor((cnn_input_size-nw)/2))
		y2 = y1 + nw - 1
		pad_img[{ {}, {}, {y1, y2} }] = ann_img
	end
	return pad_img
end
-- foward input to Normalize(2) to nn.CMul(d) with init_norm
function net_utils.normalize_scale(d, init_norm)
	local scaler = nn.CMul(d)
	scaler.weight:fill(init_norm)
	local net = nn.Sequential():add(nn.Normalize(2)):add(scaler)
	return net
end
-- kaiming initialization
function net_utils.w_init_kaiming(L)
	L:reset(math.sqrt(4/(L.weight:size(2) + L.weight:size(1))))
	L.bias:zero()
end
--[[
Combine visual representation from VisualEncoder and EmbeddingLayer
We will multiply vis_emb_feats with some constant, i.e., 20~30 to balance the feats
for simplictiy we use the opt.init_norm whose default is 20.
Input : vis_enc_feats, vis_emb_feats
Output: joined {vis_enc_feats, dropout(scaled(vis_emb_feats))} 
]]
function net_utils.VisualCombiner(opt)
	local vis_drop_out = utils.getopt(opt, 'vis_drop_out')

	local inputs = {nn.Identity()(), nn.Identity()()}
	local vis_enc_feats = inputs[1]
	local vis_emb_feats = inputs[2]
	-- scale vis_emb_feats by 20 times, empirically this value
	local scaler = net_utils.normalize_scale(opt.embedding_size, opt.init_norm)
	vis_emb_feats = scaler(vis_emb_feats)
	-- drop out vis_emb_feats (we already drop out vis_enc_feats previously)
	if vis_drop_out > 0 then vis_emb_feats = nn.Dropout(vis_drop_out)(vis_emb_feats) end
	-- join these two features
	local output = nn.JoinTable(2){vis_enc_feats, vis_emb_feats}
	-- return
	return nn.gModule(inputs, {output})
end
--[[
Build visual encoder
input : {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}
output: fc8 -> normalization -> join together
]]
function net_utils.VisualEncoder(cnn, opt)
	-- options
	local vis_use_bn = utils.getopt(opt, 'vis_use_bn')
	local use_ann = utils.getopt(opt, 'use_ann', 1)  -- 0.none, 1.vgg, 2.att, 3.frc
	local use_context = utils.getopt(opt, 'use_context', 1) -- 0.none, 1.img, 2.window2, 3.window3
	local use_location = utils.getopt(opt, 'use_location', 1)  -- 0.none, 1.use location
	local dif_ann = utils.getopt(opt, 'dif_ann', 1)  -- 0.none, 1.mean, 2.max, 3.min, 4.weighted
	local dif_location = utils.getopt(opt, 'dif_location', 1)  -- 0.none, 1.use location
	local dif_num = utils.getopt(opt, 'dif_num', 5)  -- how many objects do we need to compute dif_lfeats
	local init_norm = utils.getopt(opt, 'init_norm', 20)
	local vis_encoding_size = utils.getopt(opt, 'vis_encoding_size')
	local vis_drop_out = utils.getopt(opt, 'vis_drop_out')
	local fc8 = cnn:get(39)

	local M = nn.ParallelTable()
	local dim = 0
	-- context path
	if use_context > 0 then
		local cxt_view
		if vis_use_bn > 0 then
			cxt_view = nn.Sequential():add(fc8:clone()):add(nn.BatchNormalization(1000))
		else
			cxt_view = nn.Sequential():add(fc8:clone()):add(net_utils.normalize_scale(1000, init_norm))
		end
		M:add(cxt_view)
		dim = dim + 1000
	end
	-- region path
	if use_ann > 0 then
		local ann_view
		if vis_use_bn > 0 then
			ann_view = nn.Sequential():add(fc8:clone()):add(nn.BatchNormalization(1000))
		else
			ann_view = nn.Sequential():add(fc8:clone()):add(net_utils.normalize_scale(1000, init_norm))
		end
		M:add(ann_view)
		dim = dim + 1000
	end
	-- location
	if use_location > 0 then
		local location_view
		if vis_use_bn > 0 then
			location_view = nn.BatchNormalization(5)
		else
			location_view = net_utils.normalize_scale(5, init_norm)
		end
		M:add(location_view)
		dim = dim + 5
	end
	-- ann difference view
	if dif_ann > 0 then
		local dif_ann_view = nn.Sequential()
		if vis_use_bn > 0 then
			dif_ann_view:add(fc8:clone()):add(nn.BatchNormalization(1000))
		else
			dif_ann_view:add(fc8:clone()):add(net_utils.normalize_scale(1000, init_norm))
		end
		M:add(dif_ann_view)
		dim = dim + 1000
	end
	-- location location view
	if dif_location > 0 then
		local dif_location_view = nn.Sequential()
		if vis_use_bn > 0 then
			dif_location_view:add(nn.BatchNormalization(5*dif_num))
		else
			dif_location_view:add(net_utils.normalize_scale(5*dif_num, init_norm))
		end
		M:add(dif_location_view)
		dim = dim + 5*dif_num
	end
	-- finalize the output
	local jemb_part = nn.Sequential()
	if vis_use_bn > 0 then
		jemb_part:add(M)
				 :add(nn.JoinTable(2))
				 :add(nn.Linear(dim, vis_encoding_size))
				 :add(nn.BatchNormalization(vis_encoding_size))
	else
		-- jointly embed layer
		local J = nn.Linear(dim, vis_encoding_size)
		net_utils.w_init_kaiming(J)  -- good initialization
		jemb_part:add(M)
				 :add(nn.JoinTable(2))
				 :add(J)
	end
	-- drop out
	if vis_drop_out > 0 then jemb_part:add(nn.Dropout(vis_drop_out)) end
	return jemb_part
end
--[[
Build Language Encoder, require 'rnn'
input:  expression codes: (seq_length, n) padded with zeros in the begining
output: hidden output: (n, hidden_size)
Note, lookup_table was initialized (vocab_size+1, word_embedding) with zero mask,
thus it includes both 0 and Mp1.
It should be called as nn.LookupTableMaskZero(vocab_size+1, word_encoding_size).
]]
function net_utils.LangEncoder(lookup_table, opt)
	-- options
	local word_encoding_size = utils.getopt(opt, 'word_encoding_size')
	local word_drop_out = utils.getopt(opt, 'word_drop_out')
	local lang_encoding_size = utils.getopt(opt, 'lang_encoding_size')

	-- lookup table
	local enc = nn.Sequential()
	enc:add(lookup_table:clone('weight', 'gradWeight'))
	assert(lookup_table.weight:size(2) == word_encoding_size)

	-- drop out
	if word_drop_out > 0 then
		enc:add(nn.Dropout(word_drop_out))
	end
	
	-- core lstm
	local lstmLayer = nn.SeqLSTM(word_encoding_size, lang_encoding_size)
	lstmLayer:maskZero()
	enc:add(lstmLayer)

	-- extract last time hidden
	enc:add(nn.Select(1, -1))

	-- add normalization layer (don't know why, but empiricaly it works better)
	enc:add(nn.BatchNormalization(lang_encoding_size))
	return enc
end
--[[
Options:
- vis_encoding_size
- lang_encoding_size
- embedding_size
Input:
- visual_input    : (n, vis_encoding_size)
- lang_input      : (n, lang_encoding_size)
Output:
- cossim 		  : (n, ) range from -1 to 1
- visual_embedding: (n, embedding_size)
]]
function net_utils.cca_embedding(opt)

	local lang_encoding_size = utils.getopt(opt, 'lang_encoding_size')
	local vis_encoding_size = utils.getopt(opt, 'vis_encoding_size')
	local embedding_size = utils.getopt(opt, 'embedding_size')
	local embedding_drop_out = utils.getopt(opt, 'embedding_drop_out', 0.1)

	-- encode both view
	local V1 = nn.Linear(vis_encoding_size, embedding_size); net_utils.w_init_kaiming(V1)
	local V2 = nn.Linear(embedding_size, embedding_size); net_utils.w_init_kaiming(V2)
	local L1 = nn.Linear(lang_encoding_size, embedding_size); net_utils.w_init_kaiming(L1)
	local L2 = nn.Linear(embedding_size, embedding_size); net_utils.w_init_kaiming(L2)
	local jemb = nn.ParallelTable()
					-- visual path
					:add(nn.Sequential()
							-- :add(nn.Dropout(embedding_drop_out))
							:add(V1)
							:add(nn.BatchNormalization(embedding_size))
							:add(nn.ReLU(true))
							:add(nn.Dropout(embedding_drop_out))
							:add(V2)
							:add(nn.BatchNormalization(embedding_size))
							:add(nn.Normalize(2)))
					-- language path
					:add(nn.Sequential()
							-- :add(nn.Dropout(embedding_drop_out))
							:add(L1)
							:add(nn.BatchNormalization(embedding_size))
							:add(nn.ReLU(true))
							:add(nn.Dropout(embedding_drop_out))
							:add(L2)
							:add(nn.BatchNormalization(embedding_size))
							:add(nn.Normalize(2)))
	-- {cossim, vis_emb}
	local out2 = nn.ConcatTable():add(nn.DotProduct()) -- (n, 1)
								 :add(nn.SelectTable(1))   
	-- {cossim, visemb}
	local output = nn.Sequential()
						:add(jemb)
						:add(out2)
	return output
end
--[[
Options: encoding_size1, encoding_size2, embedding_size
Input:
- encoding_input1 : (n, encoding_size1)
- encoding_input2 : (n, encoding_size2)
Output:
- LR score: (n, )
]]
function net_utils.metric_net(opt)
	local encoding_size1 = utils.getopt(opt, 'encoding_size1')
	local encoding_size2 = utils.getopt(opt, 'encoding_size2')
	local embedding_size = utils.getopt(opt, 'embedding_size')
	local embedding_drop_out = utils.getopt(opt, 'embedding_drop_out')

	local fc1 = nn.Linear(encoding_size1+encoding_size2, embedding_size); net_utils.w_init_kaiming(fc1)
	local fc2 = nn.Linear(embedding_size, embedding_size); net_utils.w_init_kaiming(fc2)
	local fc3 = nn.Linear(embedding_size, 1); net_utils.w_init_kaiming(fc2)

	local joined = nn.Sequential()
						:add(nn.ParallelTable()
								:add(nn.BatchNormalization(encoding_size1))  -- normalize input1
								:add(nn.BatchNormalization(encoding_size2))  -- normalize input2
							)
						:add(nn.JoinTable(2))
	local metric = nn.Sequential()
						:add(fc1)
						:add(nn.BatchNormalization(embedding_size))
						:add(nn.ReLU(true))
						:add(nn.Dropout(embedding_drop_out))
						:add(fc2)
						:add(nn.BatchNormalization(embedding_size))
						:add(nn.ReLU(true))
						:add(nn.Dropout(embedding_drop_out))
						:add(fc3)
						:add(nn.Sigmoid())
	local net = nn.Sequential():add(joined):add(metric):add(nn.View(-1))
	return net
end
--[[
Extract {visemb, lang_emb} given protos, which includes 
vis_encoder, lang_encoder, cca_embedding, ...
Note, never update the two extracted modules!!! We don't provide protections here.
]]
function net_utils.extract_sub_embedding(protos)
	local dtype = protos.vis_encoder:type()
	-- construct vis_emb
	local vis_enc_part = protos.vis_encoder
	local vis_emb_part = protos.cca_embedding:get(1):get(1)
	local vis_embedding = nn.Sequential():type(dtype)
	vis_embedding:add(vis_enc_part):add(vis_emb_part)
	-- construct lang_emb
	local lang_enc_part = protos.lang_encoder
	local lang_emb_part = protos.cca_embedding:get(1):get(2)
	local lang_embedding = nn.Sequential():type(dtype)
	lang_embedding:add(lang_enc_part):add(lang_emb_part)
	-- return
	local sub_embedding = {visemb = vis_embedding, langemb = lang_embedding}
	return sub_embedding
end
--[[
Every feats is {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}
output {combined_cxt_feats, ..., }
]]
function net_utils.combine_feats(Feats)
	local nfeats = #Feats    -- number of feats
	local ntypes = #Feats[1] -- number of feat types, e.g., 5 in this case
	-- combine
	local comb_feats = {}
	for k = 1, ntypes do
		local feats_set = {}
		for i = 1, nfeats do
			table.insert(feats_set, Feats[i][k])
		end
		comb_feats[k] = torch.cat(feats_set, 1)
	end
	return comb_feats
end
--[[
Compute img_cossim of anns and sents for each image, 
later on we would use torch.multinomial() to sample hard_ann_id and hard_sent_id.
Graph will saved as cache/graphs/dataset/model_id_graphs.json

Currently, we only support hard_ann_id mining...TODO
]]
function net_utils.make_graphs(sub_embedding, loader, split, opt)

	assert(split == 'train', 'We only mine hard negatives from training split.')
	
	-- prepare models
	local visemb  = sub_embedding.visemb
	local MM = nn.MM(false, true):type(visemb:type())

	-- set to evaluate, don't forget
	visemb:evaluate()

	-- make graph = [{image_id, ann_ids, cossim}]
	local graphs = {}
	
	loader:resetImageIterator(split)
	while true do
		-- fetch data for one image
		local data = loader:getImageBatch(split, opt)
		local img_ann_ids = data.img_ann_ids
		local image_id = data.image_id
		local feats = data.feats  
		if opt.gpuid >= 0 then
			for k = 1, #feats do feats[k] = feats[k]:cuda() end
		end	
		assert(feats[1]:size(1) == #img_ann_ids)

		-- forward feats to visemb
		local vis_emb_feats = visemb:forward(feats)

		-- compute cossim (#img_ann_ids, #img_ann_ids)
		local cossim = MM:forward{vis_emb_feats, vis_emb_feats}
		cossim = cossim:float()
		cossim = utils.tensor_to_table(cossim)

		-- add to graphs
		table.insert(graphs, {image_id=image_id, ann_ids=img_ann_ids, cossim=cossim})

		-- print
		local ix0 = data.bounds.it_pos_now - 1
		local ix1 = data.bounds.it_max
		print(string.format('constructing img_to_cossim for split [%s]... %d/%d done.',
			split, ix0, ix1))

		-- jump out
		if data.bounds.wrapped then break end
	end

	-- save temp graphs
	local graph_dir = path.join('cache/graphs/', opt.dataset)
	local graph_path = path.join(graph_dir, opt.id .. '_graphs.json')
	if not utils.file_exists('cache/graphs') then os.execute('mkdir cache/graphs') end
	if not utils.file_exists(graph_dir) then os.execute('mkdir ' .. graph_dir) end
	utils.write_json(graph_path, graphs)

	-- assign graphs to loader
	loader:load_graph(graph_path)
end
-- clone a model table
function net_utils.clone_list(lst)
	-- takes list of tensors, clone all
	local new = {}
	for k,v in pairs(lst) do
		new[k] = v:clone()
	end
	return new
end

return net_utils












