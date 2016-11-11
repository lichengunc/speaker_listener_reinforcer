require 'nn'
--[[
Denotation:
- paired   : logprobs for P(Sentence | Object) of size (D+1, N, M+1)
- unpaired : logprobs for P(Sentence | Object) of size (D+1, N, M+1)

inputs : {paired, unpaired}
seqs   : {seq1, seq2}, where each seq is of size (D, N)

This criterion computes margin loss of paired logP over unpaired logP.
Paired means pair of (referring expression, referred object)
Unpaired means pair of (referring expression, other object) or pair of 
(other object, referring expression)
We hope the paired logP is higher than unpaired logP.
]]
local crit, parent = torch.class('nn.TripletRankingCriterion', 'nn.Criterion')
function crit:__init(margin)
	parent.__init(self)
	self.margin = margin or 1
end

function crit:updateOutput(inputs, seqs)
	assert(type(input) == 'table')
	assert(type(seqs) == 'table')
	-- inputs
	local paired = inputs[1]
	local unpaired = inputs[2]
	local L, N, Mp1 = paired:size(1), paired:size(2), paired:size(3)
	-- seqs
	local seq1 = seqs[1]
	local seq2 = seqs[2]
	local D, N = seq1:size(1), seq1:size(2)
	-- compute logprobs for each sent
	self.gradInput = {
					  torch.zeros(L, N, Mp1):typeAs(paired), 
					  torch.zeros(L, N, Mp1):typeAs(unpaired)
					}
	-- compute sentLogprobs
	-- 1st row is logprobs for paired 
	-- 2nd row is logprobs for unpaired
	local sentLogProbs = torch.zeros(2, N)
	-- 1st row of sentLogProbs
	for b = 1, N do
		local first_time = true
		local n = 0
		for t = 1, L do
			-- fetch the index of the token
			local target_index
			if t > D then
				target_index = 0
			else
				target_index = seq1[{t, b}]
			end
			if target_index == 0 and first_time then 
				target_index = Mp1
				first_time = false
			end
			if target_index ~= 0 then
				n = n+1
				sentLogProbs[1][b] = sentLogProbs[1][b] + paired[{ t, b, target_index }]
				self.gradInput[1][{ t, b, target_index }] = 1
			end
		end
		sentLogProbs[1][b] = sentLogProbs[1][b]/n
		self.gradInput[1][{ {}, b, {} }]:div(n)
	end
	-- 2nd row of sentLogProbs
	for b = 1, N do
		local first_time = true
		local n = 0
		for t = 1, L do
			-- fetch the index of the token
			local target_index
			if t > D then
				target_index = 0
			else
				target_index = seq2[{t, b}]
			end
			if target_index == 0 and first_time then
				target_index = Mp1
				first_time = false
			end
			if target_index ~= 0 then
				n = n+1
				sentLogProbs[2][b] = sentLogProbs[2][b] + unpaired[{ t, b, target_index }]
				self.gradInput[2][{ t, b, target_index }] = 1
			end
		end
		sentLogProbs[2][b] = sentLogProbs[2][b]/n
		self.gradInput[2][{ {}, b, {} }]:div(n)
	end
	-- compute Ranking loss for each pair
	self._output = -sentLogProbs[1]:clone()  -- of size N
	self._output:add(1, sentLogProbs[2])
	self._output:add(self.margin)
	-- compute output
	self.output = self._output:clone()
	self.output:cmax(0)  -- max(0, margin + neg_prob - pos_prob)
	self.output = self.output:sum()/N  -- average ranking loss across each pair
	return self.output
end

function crit:updateGradInput(inputs, seqs)
	-- get batch_size (of positive data)
	local N = seqs[1]:size(2) 
	-- compute mask
	-- 0 if ranking loss < 0, otherwise 1
	local mask = self._output:clone() -- of size N
	mask:ge(self._output, 0)
	mask:div(N)  -- average by total words within this seq batch

	local dsentLogProbs = torch.zeros(2, N)
	dsentLogProbs[1] = -mask
	dsentLogProbs[2] = mask

	for b=1, N do
		-- backprop on gradInput (positive half)
		if dsentLogProbs[1][b]~=0 then
			self.gradInput[1][{ {}, b, {} }]:mul(dsentLogProbs[1][b])
		else
			self.gradInput[1][{ {}, b, {} }]:zero()
		end
		-- backprop on gradInput (negative half)
		if dsentLogProbs[2][b]~=0 then
			self.gradInput[2][{ {}, b, {} }]:mul(dsentLogProbs[2][b])
		else
			self.gradInput[2][{ {}, b, {} }]:zero()
		end
	end

	return self.gradInput
end


















