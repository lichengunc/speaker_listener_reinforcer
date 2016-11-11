require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.modules.LSTM'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')

function layer:__init(lookup_table, opt)
	parent.__init(self)

	-- options for core net
	self.vocab_size = utils.getopt(opt, 'vocab_size')
	self.vis_encoding_size = utils.getopt(opt, 'vis_encoding_size')
	self.word_encoding_size = utils.getopt(opt, 'word_encoding_size')
	self.rnn_size = utils.getopt(opt, 'rnn_size')
	local dropout = utils.getopt(opt, 'dropout', 0.5)
	-- options for language model
	self.seq_length = utils.getopt(opt, 'seq_length')
	-- create core LSTM
	self.core = LSTM.lstm(self.vis_encoding_size, self.word_encoding_size, self.vocab_size+1, self.rnn_size, dropout)
	-- Note, lookup_table was initialized (vocab_size+1, word_embedding) with zero mask
	-- thus it includes both 0 and Mp1.
	-- 0 is used in LangEncoder, and Mp1 is used in LanguageModel
	self.lookup_table = lookup_table:clone('weight', 'gradWeight')
	assert(self.lookup_table.weight:size(1) == self.vocab_size+2, '+2 for 0 and Mp1')
	-- initialize
	self:_createInitState(1)
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
	print('constructing clones inside the LanguageModel')
	self.clones = {self.core}
	self.lookup_tables = {self.lookup_table}
	for t = 2, self.seq_length+1 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
	end
end

function layer:parameters()
	-- we only return self.core's parameters
	local params, grad_params = {}, {}
	local p1,g1 = self.core:parameters()
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	return params, grad_params
end

function layer:getModulesList()
	return {self.core}
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
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, opt)
	local sample_max = utils.getopt(opt, 'sample_max', 1)
	local beam_size = utils.getopt(opt, 'beam_size', 1)
	local temperature = utils.getopt(opt, 'temperature', 1.0)
	if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- direct to beam search

	local batch_size = imgs:size(1)
	self:_createInitState(batch_size)
	local state = self.init_state

	-- we will write output predictions into tensor seq
	local seq = torch.LongTensor(self.seq_length, batch_size):zero()
	local seqLogProbs = torch.FloatTensor(self.seq_length, batch_size)
	local logprobs -- logprobs predicted in last time step
	for t = 1, self.seq_length + 1 do

		local wt, it, sampleLogprobs
		if t == 1 then
			-- feed in images and START token
			it = torch.LongTensor(batch_size):fill(self.vocab_size+1)  -- START token
			wt = self.lookup_table:forward(it)
		else
			-- take predictions from previous time step and feed them in
			if sample_max == 1 then
				-- use argmax "sampling"
				sampleLogprobs, it = torch.max(logprobs, 2)
				it = it:view(-1):long()
			else
				-- sample from the distribution of previous predictions
				local prob_prev
				if temperature == 1.0 then 
					prob_prev = torch.exp(logprobs)
				else
					-- scale logprobs by temperature
					prob_prev = torch.exp(torch.div(logprobs, temperature))
				end
				it = torch.multinomial(prob_prev, 1)
				sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
				it = it:view(-1):long() -- and flatten indices for downstream processing
			end
			wt = self.lookup_table:forward(it)
		end

		if t >= 2 then
			seq[t-1] = it  -- record the samples
			seqLogProbs[t-1] = sampleLogprobs:view(-1):float()  -- and also their log likelihoods
		end

		local inputs = {imgs, wt, unpack(state)}  -- imgs and word-embedding are feed into rnn at each timestep.
		local out = self.core:forward(inputs)
		logprobs = out[self.num_state + 1]  -- last element is the output vector
		logprobs[{ {}, self.vocab_size }] = -1e5  -- make UNK very low
		state = {}
		for i = 1, self.num_state do table.insert(state, out[i]) end
	end

	-- return the samples and their log likelihoods
	return seq, seqLogProbs
end
--[[
Implements beam search
]]
function layer:sample_beam(imgs, opt)
	local beam_size = utils.getopt(opt, 'beam_size', 10)
	local batch_size, feat_dim = imgs:size(1), imgs:size(2)
	local function compare(a,b) return a.p > b.p end -- used downstream
	local function compare_ppl(a, b) return a.ppl < b.ppl end  -- used upstream

	assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

	local seq = torch.LongTensor(self.seq_length, batch_size):zero()
	local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
	local Done_beams = {}  -- will contain k done_beams
	-- lets process every image independently for now, for simplicity
	for k=1,batch_size do

		-- create initial states for all beams
		self:_createInitState(beam_size)
		local state = self.init_state

		-- we will write output predictions into tensor seq
		local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
		local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
		local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
		local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
		local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
		local done_beams = {} -- done_beams for the k-th img
		for t=1,self.seq_length+1 do

			local xt, it, sampleLogprobs
			local new_state
			if t == 1 then
				-- feed in the start tokens
				it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
				xt = self.lookup_table:forward(it)
			else
				--[[
				  perform a beam merge. that is,
				  for every previous beam we now many new possibilities to branch out
				  we need to resort our beams to maintain the loop invariant of keeping
				  the top beam_size most likely sequences.
				]]--
				local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
				ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
				local candidates = {}
				local cols = math.min(beam_size,ys:size(2))
				local rows = beam_size
				if t == 2 then rows = 1 end -- at first time step only the first beam is active
				for c=1,cols do -- for each column (word, essentially)
					for q=1,rows do -- for each beam expansion
						-- compute logprob of expanding beam q with word in (sorted) position c
						local local_logprob = ys[{ q,c }]
						local candidate_logprob = beam_logprobs_sum[q] + local_logprob
						table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
					end
				end
				table.sort(candidates, compare) -- find the best c,q pairs

				-- construct new beams
				new_state = net_utils.clone_list(state)
				local beam_seq_prev, beam_seq_logprobs_prev
				if t > 2 then
					-- well need these as reference when we fork beams around
					beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
					beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
				end
				for vix=1,beam_size do
					local v = candidates[vix]
					-- fork beam index q into index vix
					if t > 2 then
						beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
						beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
					end
					-- rearrange recurrent states
					for state_ix = 1,#new_state do
						-- copy over state in previous beam q to new beam at vix
						new_state[state_ix][vix] = state[state_ix][v.q]
					end
					-- append new end terminal at the end of this beam
					beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
					beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
					beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

					if v.c == self.vocab_size+1 or t == self.seq_length+1 then
						-- END token special case here, or we reached the end.
						-- add the beam to a set of done beams
						table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
												  logps = beam_seq_logprobs[{ {}, vix }]:clone(),
												  p = beam_logprobs_sum[vix],
												  -- ppl = -beam_logprobs_sum[vix]/(t-1)
												  ppl = torch.exp(-beam_logprobs_sum[vix]/(t-1))  -- ppl = (p1p2..pn)^(-1/n)
												 })
						-- we won't consider this beam any more
						-- otherwise, some beams would remain the top even they arrived <end>
						beam_logprobs_sum[vix] = -1000
					end
				end
				
				-- encode as vectors
				it = beam_seq[t-1]
				xt = self.lookup_table:forward(it)
			end

			if new_state then state = new_state end -- swap rnn state, if we reassinged beams

			local inputs = {imgk, xt,unpack(state)}
			local out = self.core:forward(inputs)
			logprobs = out[self.num_state+1] -- last element is the output vector
			logprobs[{ {}, self.vocab_size }] = -1e5  -- make UNK very low
			state = {}
			for i=1,self.num_state do table.insert(state, out[i]) end
		end

		-- table.sort(done_beams, compare)
		table.sort(done_beams, compare_ppl)
		seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
		seqLogprobs[{ {}, k }] = done_beams[1].logps

		-- chunk done_beams to beam_size
		for j=1+beam_size,#done_beams do table.remove(done_beams, 1+beam_size) end  
		table.insert(Done_beams, done_beams)
	end

	-- return the samples and their log likelihoods
	return seq, seqLogprobs, Done_beams
end
--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M, where M = opt.vocab_size and D = opt.seq_length

return self.output
1. a (D+1)xNx(M+1) Tensor giving (normalized) log probabilities for the
next token at every iteration of the LSTM (+1 because START/END token shift)

we also prepare the follows:
1. self.inputs,               table of each-time input (imgs, word-embedding, hidden)
2. self.lookup_tables_inputs, table of each-time word-embedding inds
3. self.tmax,                 effective seq_length of this batch
4. self.state,                table of each-time hidden (and cell) state
]]
function layer:updateOutput(input)
	local imgs = input[1]
	local seq = input[2]
	if self.clones == nil then self:createClones() end

	assert(seq:size(1) == self.seq_length)
	local batch_size = seq:size(2)

	-- intermediates
	self:_createInitState(batch_size)
	self.state = {[0] = self.init_state}
	self.inputs = {}
	self.lookup_tables_inputs = {}
	self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficienty

	-- go through rnn (seq_length+1) times
	local logprobs = torch.Tensor(self.seq_length+1, batch_size, self.vocab_size+1):typeAs(imgs)
	for t = 1, self.seq_length+1 do

		local can_skip = false
		local wt
		if t == 1 then
			-- feed in the start tokens
			local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
			self.lookup_tables_inputs[t] = it 
			wt = self.lookup_tables[t]:forward(it)  -- NxK sized input (word embedding vectors)
		else
			-- feed in the rest of the sequence
			local it = seq[t-1]:clone()
			if torch.sum(it) == 0 then can_skip = true end  -- if no effective words, we can skip from now on
			--[[
			seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
			that won't make lookup_table crash with an error.
			token #1 will do, arbitrarily. This will be ignored anyway
			because we will carefully set the loss to zero at these places
			in the criterion, so computation based on this value will be noop for the optimization.
			--]]
			it[torch.eq(it, 0)] = 1

			if not can_skip then
				self.lookup_tables_inputs[t] = it
				wt = self.lookup_tables[t]:forward(it)
			end
		end

		if not can_skip then
			-- construct the inputs
			self.inputs[t] = {imgs, wt, unpack(self.state[t-1])}
			-- forward the network
			local out = self.clones[t]:forward(self.inputs[t])
			-- process the outputs
			logprobs[t] = out[self.num_state+1]  -- last element is the output vector
			self.state[t] = {}
			for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
			self.tmax = t
		end
	end
	self.output = logprobs
	return self.output
end
--[[
gradOutput is an (D+1)xNx(M+1) Tensor.
]]
function layer:updateGradInput(input, gradOutput)
	local dimgs = torch.Tensor() -- grad on input images
	dimgs = dimgs:typeAs(input[1]):resizeAs(input[1]):zero()

	local dlogprobs = gradOutput

	-- go backwards and lets compute gradients
	local dstate = {[self.tmax] = self.init_state}  -- this works when init_state is all zeros
	for t = self.tmax, 1, -1 do
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

	-- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
	self.gradInput = {dimgs, torch.Tensor()}
	return self.gradInput
end
----------------------------------------------------------------------------------------------- 
-- addiontal components: LanguageModelCriterion, computeLosses
----------------------------------------------------------------------------------------------- 
--[[
input: Tensor of size (D+1, N, M+1) or (D+1) x 2N x (M+1) or (D+1) x 3N x (M+1)
		depending on whether we feed in [paired logprobs] or [paired, vis_unpaired, lang_unpaird]
seq: LongTensor of size (D, N), denoting labels for N referred objects (paired part of input)

We only compute the generation loss for [pos] logprobs, as [seq] and [pos] are paired.
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
	parent.__init(self)
end
function crit:updateOutput(input, seq)
	self.gradInput:resizeAs(input):zero() -- reset to zeros
	local L, Mp1 = input:size(1), input:size(3)
	local N = seq:size(2)
	local D = seq:size(1)
	assert(D == L-1, 'input Tensor should be 1 larger in time')

	local loss = 0
	local n = 0
	for b=1, N do -- iterate over [pos] batches
		local first_time = true
		for t=1, L do -- iterate over sequence time

			-- fetch the index of the next token in the sequence
			local target_index
			if t > D then -- we are out of bounds of the index sequence: pad with null tokens
				target_index = 0
			else
				target_index = seq[{t, b}] 
			end
			-- the first time we see null token as next index, actually want the model to predict the END token
			if target_index == 0 and first_time then
				target_index = Mp1
				first_time = false
			end

			-- if there is a non-null next token, enforce loss!
			if target_index ~= 0 then
				-- accumulate loss
				loss = loss - input[{ t,b,target_index }] -- log(p)
				self.gradInput[{ t,b,target_index }] = -1
				n = n + 1
			end
		end
	end
	self.output = loss / n -- normalize by number of predictions that were made
	self.gradInput:div(n)
	return self.output
end
function crit:updateGradInput(input, seq)
	return self.gradInput
end
--[[
input : (seq_length+1, N, Mp1) float
seq   : (seq_length, N) long
output: (N, ) float, which is losses for each (input_i, seq_i)
]]
function computeLosses(input, seq)
	local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
	local D = seq:size(1)
	assert(D == L-1, 'input Tensor should be 2 larger in time')

	local losses = torch.zeros(N)
	for b=1,N do -- iterate over batches
		local n = 0
		local first_time = true
		for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

			-- fetch the index of the next token in the sequence
			local target_index
			if t > D then -- we are out of bounds of the index sequence: pad with null tokens
				target_index = 0
			else
				target_index = seq[{t,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
			end
			-- the first time we see null token as next index, actually want the model to predict the END token
			if target_index == 0 and first_time then
				target_index = Mp1
				first_time = false
			end

			-- if there is a non-null next token, enforce loss!
			if target_index ~= 0 then
				-- accumulate loss
				losses[b] = losses[b] - input[{ t,b,target_index }] -- log(p)
				n = n + 1
			end
		end
		-- normalize current sequence by number of effective words
		if n == 0 then losses[b] = 0 else losses[b] = losses[b]/n end
	end

	return losses
end
