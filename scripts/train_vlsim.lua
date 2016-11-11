--[[
This code is to train a vis-lang similarity network.
The input is a pair of {visual_feats, lang_feats}
The output is simple: yes or no! 
We will use binary cross entropy to gudie the training.
Later on, the similarity score will be regared as a reward in the reinforcement learning for a given pair.
]]
require 'torch'
require 'nn'
require 'nngraph'
-- exotic import
require 'loadcaffe'
-- local imports
require 'misc.optim_updates'
require 'misc.DataLoader'
require 'misc.modules.SplitEmbedding'
require 'misc.modules.SplitGeneration'
require 'misc.modules.TripletRankingCriterion'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local eval_utils = require 'misc.eval_utils'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
-- Data input settings
cmd:option('-dataset', 'refcoco_unc', 'name of dataset+splitBy')
cmd:option('-cnn_proto', 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model', 'models/vgg/VGG_ILSVRC_16_layers.caffemodel', 'path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
-- Visual Encoder Setting
cmd:option('-use_context', 1, '1. image, 2. window2, 3. window3, 4. window4, 5. window5')
cmd:option('-use_ann', 1, '1. use regional feature; otherwise do not use.')
cmd:option('-use_location', 1, '1. use location feature; otherwise do not use.')
cmd:option('-dif_ann', 1, 'use visual comparison')
cmd:option('-dif_location', 1, 'use location comparison')
cmd:option('-dif_source', 1, '1.st_anns, 2.dt_anns, 3.all_anns')
cmd:option('-dif_pool', 1, '1.mean, 2.max, 3.min. 4.weighted')
cmd:option('-dif_num', 5, 'number of objects needed to compute visual difference')
cmd:option('-vis_use_bn', 0, 'if use batch normalization in visual encoder')
cmd:option('-init_norm', 20, 'norm of each visual representation')
cmd:option('-vis_encoding_size', 512)
cmd:option('-vis_drop_out', 0.25)
-- Sampling Setting
cmd:option('-mine_hard', 0, 'do hard negative mining?')
cmd:option('-hard_temperature', 5, 'temperature for scaling cossim before SoftMax, larger means lower entropy')
cmd:option('-mine_hard_every', 4000, 'mine hard negatives every how many iterations')
-- Language Encoder Setting
cmd:option('-word_encoding_size', 512, 'the encoding size of each token in the vocabulary')
cmd:option('-word_drop_out', 0.5)
cmd:option('-lang_encoding_size', 512, 'hidden size of LSTM')
-- Embedding Setting
cmd:option('-embedding_size', 512, 'joint embedding layer dimension')
cmd:option('-embedding_drop_out', 0.2, 'strength of dropout in the input visual and language representation')
-- Language Model Setting
cmd:option('-rnn_size', 512, 'hidden size of LSTM')
cmd:option('-lm_drop_out', 0.5, 'strength of dropout in the Language Model RNN')
-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-sample_ratio', 0.5, 'ratio of same-type objects over different-type objects.')
cmd:option('-batch_size', 32, 'what is the batch size in number of referred objects per batch? (there will be x seq_per_ref sentences)')
cmd:option('-grad_clip', 0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-seq_per_ref', 3,'number of expressions to sample for each referred object during training.')
cmd:option('-learning_rate_decay_start', 8000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 8000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_epsilon', 1e-8,'epsilon that goes into denominator for smoothing') 
-- Optimization: for WE, LangEncoder, CCA, LM
cmd:option('-learning_rate', 4e-4,'learning rate')
cmd:option('-optim_alpha', 0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta', 0.999,'beta used for adam')
-- Optimization: for VisualEncoder
cmd:option('-ve_learning_rate', 4e-5,'learning rate for the joint embedding')
cmd:option('-ve_optim_alpha', 0.8,'alpha for momentum of joint embedding')
cmd:option('-ve_optim_beta', 0.999,'alpha for momentum of joint embedding')
-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 2000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'models/vl_metric_models', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 24, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()
-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
-- For CPU
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
-- For GPU
if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
	cutorch.manualSeed(opt.seed)
	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
print(opt)
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5   = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_json = data_json, data_h5 = data_h5}

-- load extracted features: call scripts/extract_xxx_feats before training!
local feats_dir = 'cache/feats/' .. opt.dataset
local featsOpt = {  ann = feats_dir .. '/ann_feats.h5',
					img = feats_dir .. '/img_feats.h5',
					det = feats_dir .. '/det_feats.h5',
					window2 = feats_dir .. '/window2_feats.h5',
					window3 = feats_dir .. '/window3_feats.h5',
					window4 = feats_dir .. '/window4_feats.h5',
					window5 = feats_dir .. '/window5_feats.h5' }
loader:loadFeats(featsOpt)

-- load cossim graph
local graph_path = path.join('cache', 'graphs', opt.dataset, 'ann_graphs.json')  -- a good initialization
if utils.file_exists(graph_path) then
	loader:load_graph(graph_path)
end
-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
local iter = 0
local ve_optim_state = {}
local le_optim_state = {}
local mn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local lr_history = {}
local best_score

if string.len(opt.start_from) > 0 then
else
	-- create visual encoder
	local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
	protos.vis_encoder = net_utils.VisualEncoder(cnn_raw, opt)

	-- create language encoder, which shares we's parameters and gradParams
	local we = nn.LookupTableMaskZero(loader:getVocabSize()+1, opt.word_encoding_size)  -- vocab_size+1 as 0 used in LangEncoder
	protos.lang_encoder = net_utils.LangEncoder(we, opt)

	-- create embedding layer
	protos.metric_net = net_utils.metric_net({encoding_size1 = opt.vis_encoding_size,
											  encoding_size2 = opt.lang_encoding_size,
											  embedding_size = opt.embedding_size,
											  embedding_drop_out = opt.embedding_drop_out
											  })
end

-- prepare criterion
local bce_crit = nn.BCECriterion()

-- ship to GPU
if opt.gpuid >= 0 then
	for k, v in pairs(protos) do v:cuda() end
	bce_crit:cuda()
end

-- flatten and prepare all model parameters to a single vector
local ve_params, ve_grad_params = protos.vis_encoder:getParameters()
print('total number of parameters in vis_encoder: ', ve_params:nElement())

local le_params, le_grad_params = protos.lang_encoder:getParameters()
print('total number of parameters in lang_encoder: ', le_params:nElement())

local mn_params, mn_grad_params = protos.metric_net:getParameters()
print('total number of parameters in metric_net: ', mn_params:nElement())

collectgarbage()
-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
loader:shuffle('train')
local function lossFun(iter)

	-- training mode
	for k, v in pairs(protos) do v:training() end

	-- zero gradients
	ve_grad_params:zero()
	le_grad_params:zero()
	mn_grad_params:zero()

	-- fetch data
	local data = loader:getBatch('train', opt)
	local ref_ann_ids = data.ref_ann_ids
	local pos_feats = data.feats
	local neg_feats = data.neg_feats
	local pos_zseq  = data.zseq
	local neg_zseq  = data.neg_zseq

	-- combine feats and zseq
	local feats = net_utils.combine_feats{pos_feats, neg_feats, pos_feats}
	local zseq  = torch.cat({pos_zseq, pos_zseq, neg_zseq}, 2)

	-- make labels
	local labels = torch.cat({ torch.ones(opt.batch_size*opt.seq_per_ref),   -- paired
							   torch.zeros(opt.batch_size*opt.seq_per_ref),  -- unpaired
							   torch.zeros(opt.batch_size*opt.seq_per_ref)}) -- unpaired

	-- ship to GPU if any
	if opt.gpuid >= 0 then
		for k = 1, #feats do feats[k] = feats[k]:cuda() end
		zseq = zseq:cuda()
		labels = labels:cuda()
	end

	-- forward net
	local vis_enc_feats  = protos.vis_encoder:forward(feats)
	local lang_enc_feats = protos.lang_encoder:forward(zseq)
	local score = protos.metric_net:forward{vis_enc_feats, lang_enc_feats}
	local loss = bce_crit:forward(score, labels)

	-- backward net
	local dscore = bce_crit:backward(score, labels)
	local dvis_enc_feats, dlang_enc_feats = unpack(protos.metric_net:backward({vis_enc_feats, lang_enc_feats}, dscore))
	protos.lang_encoder:backward(zseq, dlang_enc_feats)
	protos.vis_encoder:backward(feats, dvis_enc_feats)

	-- shuffle the train
	-- Do not like this... as we should do the shuffle before-hand, 
	-- making CPU reading from disk sequentially, which could save IO time
	-- but for each referred object, we need to check its surrounding objects for negative mining
	-- Besides, we also need to shuffle refs. It became hard to do shuffling as preprocessing step.
	if data.bounds.wrapped then loader:shuffle('train') end -- After one epoch, let's reshuffle the data.

	-- get out
	local losses = {total_loss = loss}
	return losses
end
	
-- evaluate script
function eval_loss(split, opt)

	local verbose = utils.getopt(opt, 'verbose', true)

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
-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
while true do

	local losses = lossFun(iter)
	if iter % opt.losses_log_every == 0 then
		loss_history[iter] = losses.total_loss
		lr_history[iter] = {jemb_learning_rate = jemb_learning_rate, lm_learning_rate = lm_learning_rate}
	end
	print(string.format('dataset[%s], id[%s], gpuid[%s], iter %d: %f', opt.dataset, opt.id, opt.gpuid, iter, losses.total_loss))

	-- eval loss / gradient
	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
		-- evaluate the validation performance
		local val_loss, val_accuracy = eval_loss('val', opt)
		print('validation loss: ', val_loss)
		print('validation accuracy: ', val_accuracy)
		val_loss_history[iter] = val_loss

		-- write json report
		if not utils.file_exists(opt.checkpoint_path) then
			os.execute('mkdir ' .. opt.checkpoint_path)
		end
		if not utils.file_exists(path.join(opt.checkpoint_path, opt.dataset)) then
			os.execute('mkdir ' .. path.join(opt.checkpoint_path, opt.dataset))
		end
		local checkpoint_path = path.join(opt.checkpoint_path, opt.dataset, 'model_id' .. opt.id)
		local checkpoint = {}
		checkpoint.opt = opt
		checkpoint.iter = iter
		checkpoint.loss_history = loss_history
		checkpoint.val_loss_history = val_loss_history
		checkpoint.val_accuracy = val_accuracy -- save these too for CIDEr/METEOR/etc eval
		checkpoint.lr_history = lr_history
		utils.write_json(checkpoint_path .. '.json', checkpoint)
		print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

		-- write full model checkpoint if we did better than ever
		local current_score = -val_loss
		if best_score == nil or current_score > best_score then
			best_score = current_score
			if iter > 0 then
				-- include the protos (which have weights) and save to file
				checkpoint.protos = protos
				checkpoint.best_score = checkpoint.best_score
				checkpoint.ve_optim_state = ve_optim_state
				checkpoint.le_optim_state = le_optim_state
				checkpoint.mn_optim_state = mn_optim_state
				checkpoint.iterators = loader.iterators
				-- also include the vocabulary mapping so that we can use the checkpoint 
				-- alone to run on arbitrary images without the data loader
				checkpoint.vocab = loader:getVocab()
				torch.save(checkpoint_path .. '.t7', checkpoint)
				print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
			end
		end
	end
	-- decay the learning rates
	local ve_learning_rate = opt.ve_learning_rate
	local learning_rate = opt.learning_rate
	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
		local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
		local decay_factor = math.pow(0.1, frac)
		ve_learning_rate = ve_learning_rate * decay_factor
		learning_rate = learning_rate * decay_factor
	end

	-- optimize
	adam(ve_params, ve_grad_params, ve_learning_rate, opt.ve_optim_alpha, opt.ve_optim_beta, opt.optim_epsilon, ve_optim_state)
	adam(le_params, le_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, le_optim_state)
	adam(mn_params, mn_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, mn_optim_state)

	-- stopping criterions
	iter = iter + 1
	if iter % 500 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
	if loss0 == nil then loss0 = losses.total_loss end
	if losses.total_loss > loss0 * 20 then
		print('loss seems to be exploding, quitting.')
		break
	end
	if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion	
end































