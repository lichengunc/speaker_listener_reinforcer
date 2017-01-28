require 'torch'
require 'nn'
require 'nngraph'
-- exotic import
require 'loadcaffe'
-- local imports
require 'misc.optim_updates'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.ReinforceLanguage'
require 'misc.ListenerReward'
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
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-dataset', 'refcoco_unc', 'name of dataset+splitBy')
cmd:option('-cnn_proto', 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model', 'models/vgg/VGG_ILSVRC_16_layers.caffemodel', 'path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
-- Reinforce Setting
cmd:option('-check_sent', 0, 'check the sampled sentence during training time')
cmd:option('-temperature', 1, 'temperature controling')
cmd:option('-vl_metric_model_id', '0', 'model_id for vl_metric_model')
cmd:option('-reward_scale', 1, 'reward scaling factor')
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
cmd:option('-embedding_drop_out', 0.1, 'strength of dropout in the input visual and language representation')
-- Language Model Setting
cmd:option('-rnn_size', 512, 'hidden size of LSTM')
cmd:option('-lm_drop_out', 0.5, 'strength of dropout in the Language Model RNN')
-- Loss Setting
cmd:option('-generation_weight', 1, 'always make weight on generation loss = 1')
cmd:option('-vis_rank_weight',  1, 'Generation: weight on paired (ref, sent) over unpaired (other object, sent)')
cmd:option('-lang_rank_weight', 0, 'Generation: weight on paired (ref, sent) over unpaired (ref, other sent)')
cmd:option('-embedding_weight', 1, 'Embedding : weight on both vis_rank and lang_rank for embedding model')
cmd:option('-lm_margin', 1, 'margin for LM ranking loss')
cmd:option('-emb_margin', 0.1, 'margin for embedding ranking loss')
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
cmd:option('-checkpoint_path', 'models', 'folder to save checkpoints into (empty = this folder)')
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
local iter = 1
local ve_optim_state  = {}
local we_optim_state  = {}
local le_optim_state  = {}
local cca_optim_state = {}
local lm_optim_state  = {}
local bs_optim_state  = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local lr_history = {}
local best_score

if string.len(opt.start_from) > 0 then
	local checkpoint = torch.load(opt.start_from)
	protos = checkpoint.protos
else
	-- create visual encoder
	local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
	protos.vis_encoder = net_utils.VisualEncoder(cnn_raw, opt)
		
	-- create word embedding of size vocab_size+2, 0 used in LangEncoder and Mp1 used in LanguageModel
	protos.we = nn.LookupTableMaskZero(loader:getVocabSize()+1, opt.word_encoding_size)

	-- create language encoder
	protos.lang_encoder = net_utils.LangEncoder(protos.we, opt)

	-- create embedding layer
	protos.cca_embedding = net_utils.cca_embedding(opt)

	-- create visual combiner
	protos.vis_combiner = net_utils.VisualCombiner(opt)

	-- create language model
	local lmOpt = { vocab_size = loader:getVocabSize(),
					vis_encoding_size = opt.vis_encoding_size + opt.embedding_size,
					word_encoding_size = opt.word_encoding_size,
					rnn_size = opt.rnn_size,
					dropout = opt.lm_drop_out,
					seq_length = loader:getSeqLength()
				  }
	protos.lm = nn.LanguageModel(protos.we, lmOpt)

	-- create split flow
	protos.split_emb = nn.SplitEmbedding(true, true)
	protos.split_lm  = nn.SplitGeneration(true, true)
end

-- prepare parallel criterions
local lm_crits = {}
local lm_crits = nn.ParallelCriterion()
lm_crits:add(nn.LanguageModelCriterion(), opt.generation_weight) -- generation path
lm_crits:add(nn.TripletRankingCriterion(opt.lm_margin), opt.vis_rank_weight)  -- vis rank path
lm_crits:add(nn.TripletRankingCriterion(opt.lm_margin), opt.lang_rank_weight) -- lang rank path

local emb_crits = nn.ParallelCriterion()
emb_crits:add(nn.MarginRankingCriterion(opt.emb_margin), opt.embedding_weight)  -- vis rank path
emb_crits:add(nn.MarginRankingCriterion(opt.emb_margin), opt.embedding_weight)  -- lang rank path

-- ship everything to GPU
if opt.gpuid >= 0 then
	for k, v in pairs(protos) do v:cuda() end
	lm_crits:cuda()
	emb_crits:cuda()
end

-- flatten and prepare all model parameters to a single vector
local we_params, we_grad_params = protos.we:getParameters()
print('total number of parameters in word_embedding: ', we_params:nElement())

local ve_params, ve_grad_params = protos.vis_encoder:getParameters()
print('total number of parameters in vis_encoder: ', ve_params:nElement())

local le_sub_net = nn.Sequential()  -- excluding the shared lookup_table
for i = 2, protos.lang_encoder:size() do le_sub_net:add(protos.lang_encoder:get(i)) end
local le_params, le_grad_params = le_sub_net:getParameters()
print('total number of parameters in lang_encoder: ', le_params:nElement())

local cca_params, cca_grad_params = protos.cca_embedding:getParameters()
print('total number of parameters in cca_embedding: ', cca_params:nElement())

local lm_params, lm_grad_params = protos.lm:getParameters()
print('total number of parameters in LM: ', lm_params:nElement())

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

-- sub_embedding = {visemb, langemb}, 
-- we will use it to construct img's cossim for hard negative mining
local sub_embedding = net_utils.extract_sub_embedding(protos)
-------------------------------------------------------------------------------
-- Create Reinforce Parts
-------------------------------------------------------------------------------
-- create reinforce parts
local rl_lm = nn.ReinforceLanguage(protos.lm, opt.temperature)
rl_lm:evaluate()  -- we want this to be more deterministic

-- load pretrained vl_metric_model
local vl_metric_model_path = path.join('models/vl_metric_models', opt.dataset, 'model_id' .. opt.vl_metric_model_id .. '.t7')
local vl_metric_model = torch.load(vl_metric_model_path).protos
local rl_crit = nn.ListenerReward(rl_lm, vl_metric_model, opt.reward_scale)

-- baseline
local adder = nn.Add(1); adder.bias:zero()
local rl_baseline = nn.Sequential():add(nn.Constant(0.5, 1)):add(adder)

-- ship to GPU
if opt.gpuid >= 0 then
	rl_lm:cuda()
	rl_crit:cuda()
	rl_baseline:cuda()
end

-- share cores and get parameters after type convert
rl_lm:shareCores(protos.lm)  -- share again here as the previous :cuda() :getParameters() issue.
rl_lm:createClones()

-- create baseline net
local bs_params, bs_grad_params = rl_baseline:getParameters()  -- do this after :cuda()
print('total number of parameters in BS: ', bs_params:nElement())

collectgarbage()
-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
loader:shuffle('train')
local function lossFun(iter)
	
	-- training mode
	for k, v in pairs(protos) do v:training() end
	
	-- zero gradients
	we_grad_params:zero()
	ve_grad_params:zero()
	le_grad_params:zero()
	cca_grad_params:zero()
	lm_grad_params:zero()
	bs_grad_params:zero()

	-- fetch data
	local data = loader:getBatch('train', opt)
	local ref_ann_ids = data.ref_ann_ids
	local pos_feats = data.feats
	local neg_feats = data.neg_feats
	local pos_seqz  = data.seqz
	local pos_zseq  = data.zseq
	local neg_seqz  = data.neg_seqz
	local neg_zseq  = data.neg_zseq

	-- combine feats
	local feats = net_utils.combine_feats{pos_feats, neg_feats, pos_feats}

	-- make seqz and zseq
	local seqz = torch.cat({pos_seqz, pos_seqz, neg_seqz}, 2)
	local zseq = torch.cat({pos_zseq, pos_zseq, neg_zseq}, 2)

	-- make labels
	local lm_labels  = { pos_seqz, {pos_seqz, pos_seqz}, {pos_seqz, neg_seqz} }
	local emb_labels = { torch.ones(#ref_ann_ids), torch.ones(#ref_ann_ids) }

	-- ship to GPU if any
	if opt.gpuid >= 0 then
		for k = 1, #feats do feats[k] = feats[k]:cuda() end
		zseq = zseq:cuda()
		for k = 1, #emb_labels do emb_labels[k] = emb_labels[k]:cuda() end
	end

	-- forward net
	local vis_enc_feats  = protos.vis_encoder:forward(feats) 
	local lang_enc_feats = protos.lang_encoder:forward(zseq)
	local cossim, vis_emb_feats = unpack(protos.cca_embedding:forward{vis_enc_feats, lang_enc_feats})
	local vis_feats = protos.vis_combiner:forward{vis_enc_feats, vis_emb_feats}
	local logprobs  = protos.lm:forward{vis_feats, seqz}
	-- forward reinforce part
	local sampled_seq  = rl_lm:forward(vis_feats)
	local sampled_zseq = loader:seqz_to_zseq(sampled_seq)
	sampled_zseq       = sampled_zseq:type(zseq:type())
	local baseline     = rl_baseline:forward(vis_feats)
	-- compute loss
	local emb_flows = protos.split_emb:forward(cossim)
	local emb_loss  = emb_crits:forward(emb_flows, emb_labels)
	local lm_flows  = protos.split_lm:forward(logprobs)
	local lm_loss   = lm_crits:forward(lm_flows, lm_labels)
	local rl_loss   = rl_crit:forward({feats, sampled_zseq, baseline}, torch.Tensor())
	local loss = lm_loss + emb_loss + rl_loss

	-- backward crits
	local dlm_flows  = lm_crits:backward(lm_flows, lm_labels)
	local dlogprobs  = protos.split_lm:backward(logprobs, dlm_flows)
	local demb_flows = emb_crits:backward(emb_flows, emb_labels)
	local dcossim    = protos.split_emb:backward(cossim, demb_flows)
	local _, _, dbaseline  = unpack(rl_crit:backward({feats, sampled_zseq, baseline}, torch.Tensor()))
	-- backward reinforce part
	rl_baseline:backward(vis_feats, dbaseline)
	local dvis_feats1 = rl_lm:backward(vis_feats, torch.Tensor())
	-- backward net
	local dvis_feats2, _ = unpack(protos.lm:backward({vis_feats, seqz}, dlogprobs))
	local dvis_feats = dvis_feats1 + dvis_feats2
	local dvis_enc_feats1, dvis_emb_feats = 
		unpack(protos.vis_combiner:backward({vis_enc_feats, vis_emb_feats}, dvis_feats))
	local dvis_enc_feats2, dlang_enc_feats = 
		unpack(protos.cca_embedding:backward({vis_enc_feats, lang_enc_feats}, {dcossim, dvis_emb_feats}))
	protos.lang_encoder:backward(zseq, dlang_enc_feats)
	protos.vis_encoder:backward(feats, dvis_enc_feats1 + dvis_enc_feats2)

	-- clip gradients
	lm_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	
	-- shuffle the train
	-- Do not like this... as we should do the shuffle before-hand, 
	-- making CPU reading from disk sequentially, which could save IO time
	-- but for each referred object, we need to check its surrounding objects for negative mining
	-- Besides, we also need to shuffle refs. It became hard to do shuffling as preprocessing step.
	if data.bounds.wrapped then loader:shuffle('train') end -- After one epoch, let's reshuffle the data.

	-- print
	if opt.check_sent > 0 then
		local sampled_sents = loader:decode_sequence(sampled_zseq)
		local reward = rl_lm.reward
		for n = 1, #sampled_sents do
			local ref_id = data.ref_ids[n]
			local sent_ids = loader.Refs[ref_id]['sent_ids']
			print('gd sents are: ')
			for k = 1, #sent_ids do
				local sent = loader.Sentences[sent_ids[k]]
				print(string.format('%s: %s', k, sent['sent']))
			end
			print(string.format('sampled sent: %s', sampled_sents[n]))
			print(string.format('reward is %.3f\n', reward[n]))
		end
	end

	-- get out
	local reward_acc = torch.ge(rl_crit.reward, 0.5):sum() / rl_crit.reward:nElement()
	local reward_bs  = baseline[1][1]
	local losses = {total_loss = loss, reward_acc = reward_acc, reward_bs = reward_bs}
	return losses
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
	print(string.format('dataset[%s], id[%s], gpuid[%s], iter %d: %.4f, reward: bs%.3f, acc%.2f%%', 
		opt.dataset, opt.id, opt.gpuid, iter, losses.total_loss, losses.reward_bs, losses.reward_acc*100))

	-- mine hard negatives, we escape the first 2000~4000 iterations to get "mature" network
	if (iter % opt.mine_hard_every == 0 and opt.mine_hard > 0 and iter > 1) then
		net_utils.make_graphs(sub_embedding, loader, 'train', opt)
	end

	-- eval loss / gradient
	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
		-- evaluate the validation performance
		local val_loss, val_accuracy = eval_utils.eval_loss(protos, loader, 'val', opt)
		print('validation loss: ', val_loss)
		print('validation accuracy: ', val_accuracy)
		val_loss_history[iter] = val_loss

		-- write json report
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
				checkpoint.we_optim_state = we_optim_state
				checkpoint.le_optim_state = le_optim_state
				checkpoint.cca_optim_state = cca_optim_state
				checkpoint.lm_optim_state = lm_optim_state
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
	adam(we_params, we_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, we_optim_state)
	adam(le_params, le_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, le_optim_state)
	adam(cca_params,cca_grad_params,learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, cca_optim_state)
	adam(lm_params, lm_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, lm_optim_state)
	adam(bs_params, bs_grad_params, learning_rate*0.5, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, bs_optim_state)

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





































