require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.modules.LSTM'

-------------------------------------------------------------------------------
-- Reinforce Language Model core
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.ReinforceLanguage', 'nn.Reinforce')

function layer:__init(lm, temperature)
	parent.__init(self)
	
	-- options
	self.vocab_size = lm.vocab_size
	self.seq_length = lm.seq_length
	self.rnn_size = lm.rnn_size
	self.temperature = temperature or 1 

	-- clone core and looup_table, and carefully share parameters
	self.core = lm.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
	self.lookup_table = lm.lookup_table:clone('weight', 'bias', 'gradWeight', 'gradBias')
	assert(self.lookup_table.weight:size(1) == self.vocab_size+2, '+2 for 0 and Mp1')

	-- initialize
	self:_createInitState(1)
end
--[[
In case we call lm:cuda() after this layer's initialization, which breaks the sharing relations.
We could hard-share the cores again by calling layer:shareCores(lm)
]]
function layer:shareCores(language_model)
	print('Sharing core nets from lm to RL_lm.')
	self.core = language_model.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
	self.lookup_table = language_model.lookup_table:clone('weight', 'bias', 'gradWeight', 'gradBias')
	assert(self.lookup_table.weight:size(1) == self.vocab_size + 2)
end

function layer:_createInitState(batch_size)
	assert(batch_size~=nil)
	-- construct the initial state for the LSTM
	if not self.init_state then self.init_state = {} end
	-- initialize prev_c and prev_h as all zeros
	self.num_state = 2
	for i = 1, self.num_state do
		if self.init_state[i] then 
			self.init_state[i]:resize(batch_size, self.rnn_size):zero() 
		else 
			self.init_state[i] = torch.zeros(batch_size, self.rnn_size)
		end
	end
end

function layer:createClones()
	-- construct the net clones
	print('constructing clones inside the Reinforce LanguageModel')
	self.clones = {self.core}
	self.lookup_tables = {self.lookup_table}
	for t = 2, self.seq_length+1 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
	end
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
end
--[[
input
- imgs: torch.Tensor of size NxK (K is dim of image code)
return
- seq:  (seq_length, N) 

we also prepare the follows:
1. self.inputs, 				table of each-time input (imgs, word-embedding, hidden)
2. self.lookup_tables_inputs,   table of each-time word-embedding inds
3. self.state, 					table of each-time hidden (and cell) state
4. self.dlogprobs,	 			sequence logprobs (seq_length+1, N, vocab_size+1)
]]
function layer:updateOutput(input)

	local imgs = input
	local batch_size = imgs:size(1)
	if self.clones == nil then self:createClones() end
	self:_createInitState(batch_size)
	
	self.state = {[0] = self.init_state}
	self.seq = torch.LongTensor(self.seq_length, batch_size):zero()
	self.inputs = {}
	self.lookup_tables_inputs = {}

	local logprobs = torch.Tensor(self.seq_length+1, batch_size, self.vocab_size+1):typeAs(imgs)
	for t = 1, self.seq_length+1 do

		local wt, it
		if t == 1 then
			-- feed in start tokens
			it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
			self.lookup_tables_inputs[t] = it
			wt = self.lookup_tables[t]:forward(it)
		else
			if self.stochastic or self.train ~= false then
				local prob_prev 
				if self.temperature == 1 then
					prob_prev = torch.exp(logprobs[t-1])  -- (n, vocab_size+1)
				else
					prob_prev = torch.exp(logprobs[t-1]/self.temperature)
					prob_prev:div(torch.sum(prob_prev))
				end
				it = torch.multinomial(prob_prev:float(), 1)  -- convert to float as there's bug for cuda multinomial
			else
				_, it = torch.max(logprobs[t-1], 2)
			end
			it = it:view(-1):long()  -- (n, )
			self.lookup_tables_inputs[t] = it
			wt = self.lookup_tables[t]:forward(it)
		end
		-- add to seq
		if t >= 2 then self.seq[t-1] = it end
		-- construct the inputs
		self.inputs[t] = {imgs, wt, unpack(self.state[t-1])}
		-- forward the network
		local out = self.clones[t]:forward(self.inputs[t])
		-- process the outputs
		logprobs[t] = out[self.num_state+1]  -- last element is the output vector
		self.state[t] = {}
		for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
	end

	-- convert self.seq to sents
	self.output = self.seq
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	-- Note that gradOutput is ignored
	-- S : sum_i{ y .* logp }
	-- x : logprobs
	--    d S       1    if y_i = 1
	-- --------- =   
	--  d logp_i    0    otherwise
	-- Then we compute dlogp/d_{imgs} and dlogp/d_{word_embedding}
	local batch_size = input:size(1)
	local Mp1 = self.vocab_size + 1
	local dlogprobs = torch.Tensor(self.seq_length+1, batch_size, self.vocab_size+1):zero():typeAs(input)
	local n = 0
	for b = 1, batch_size do

		local stop_sign = false
		for t = 1, self.seq_length + 1 do
			-- fetch the index of the next token
			local target_index
			if t > self.seq_length then
				target_index = Mp1
			else
				target_index = self.seq[{t, b}]
			end
			-- if not stop yet, enforce supervision
			if not stop_sign then
				assert(target_index <= dlogprobs:size(3), 
					'out of bound, target_index='..target_index..' dlogprobs:size(3) '..dlogprobs:size(3))
				dlogprobs[{ t,b,target_index }] = -1
				n = n + 1
			end
			-- if not stop yet and just saw Mp1, let's stop 
			if target_index == Mp1 and stop_sign == false then
				stop_sign = true
			end
		end
	end
	dlogprobs:div(n)

	-- Reward multiplication
	-- Note we don't call self:rewardAs because of the size issue 
	-- dlogprobs is (seq_length+1, n, Mp1), reward is (n, )
	assert(dlogprobs:size(2) == self.reward:nElement())
	for b = 1, batch_size do
		dlogprobs[{ {}, b, {} }] = dlogprobs[{ {}, b, {} }]*self.reward[b]
	end

	-- Given dlogprobs, let's backward the LSTM to imgs and word embedding
	dlogprobs = dlogprobs:type(input:type())
	local dimgs = input:clone():zero()
	local dstate = {[self.seq_length+1] = self.init_state}
	for t = self.seq_length+1, 1, -1 do
		-- concat state gradients and output vector gradients at time step t
		local dout = {}
		for k = 1, #dstate[t] do table.insert(dout, dstate[t][k]) end
		-- insert dlogprobs
		table.insert(dout, dlogprobs[t])
		-- backward
		local dinputs = self.clones[t]:backward(self.inputs[t], dout)
		-- accumulate dimgs
		dimgs = dimgs + dinputs[1]
		-- continue backprop of wt
		local dwt = dinputs[2]
		local it = self.lookup_tables_inputs[t]
		self.lookup_tables[t]:backward(it, dwt)  -- backprop into lookup table
		-- copy over rest to state grad
		dstate[t-1] = {}
		for k = 3, self.num_state+2 do table.insert(dstate[t-1], dinputs[k]) end
	end
	self.gradInput = dimgs
	return self.gradInput
end







































