require 'nn'
--[[
Denotation:
- paired 	    : logprobs for P(referring expression | referred object)
- vis_unpaired  : logprobs for P(referring expression | negative object)
- lang_unpaired : logprobs for P(other expression | referred object)

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