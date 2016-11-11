
local ListenerReward, parent = torch.class('nn.ListenerReward', 'nn.Criterion')

function ListenerReward:__init(module, vl_metric_model, scale, criterion)
	parent.__init(self)
	self.module = module
	self.scale = scale or 1
	self.vl_metric_model = vl_metric_model  -- vis_encoder, lang_encoder, metric_net
	self.criterion = criterion or nn.MSECriterion()
	self.gradInput = {torch.Tensor()}
end
--[[
input:
- feats: {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}, each is (n, dim)
- zseq : long (seq_length, n)
- baseline: (n, 1)
]]
function ListenerReward:updateOutput(input, target)

	assert(torch.type(input) == 'table')
	local feats = input[1]
	local zseq  = input[2]
	
	-- check shape
	assert(feats[1]:size(1) == zseq:size(2))
	local _size = feats[1]:size(1)

	-- check dtype
	local dtype = self.vl_metric_model.vis_encoder:type()
	assert(feats[1]:type() == dtype)
	assert(zseq:type() == dtype)

	-- set vl_metric_model to evaluate()
	for k, v in pairs(self.vl_metric_model) do
		v:evaluate()  -- set evalute mode
	end
	
	-- compute lr_score
	local vis_enc_feats  = self.vl_metric_model.vis_encoder:forward(feats) 
	local lang_enc_feats = self.vl_metric_model.lang_encoder:forward(zseq)
	local lr_score = self.vl_metric_model.metric_net:forward{vis_enc_feats, lang_enc_feats}  -- (n,) range from 0 to 1

	-- compute reward
	self.reward = lr_score
	self.reward:mul(self.scale)

	-- loss = -sum(reward)/_size
	self.output = -self.reward:sum()/_size
	return self.output
end

function ListenerReward:updateGradInput(input, target)

	local baseline = input[3]
	local _size = input[1][1]:size(1)

	-- reduce variance of reward using baseline
	self.vrReward = self.reward:clone()
	self.vrReward:add(-1, baseline)

	-- broadcast reward to modules
	self.module:reinforce(self.vrReward)

	-- zero gradInput
	self.gradInput = {}
	self.gradInput[1] = torch.Tensor()
	self.gradInput[2] = torch.Tensor()
	self.gradInput[3] = self.criterion:backward(baseline, self.reward)

	return self.gradInput
end

function ListenerReward:type(type)
	local module = self.module  -- keep unchanged
	self.module = nil
	local ret = parent.type(self, type)
	self.module = module
	-- type convert each module of vl_metric_model
	for k, v in pairs(self.vl_metric_model) do
		v:type(type)
	end
	return ret
end



























