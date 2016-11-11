require 'nn'
--[[
Denotation:
- paired 	    : logprobs for P(referring expression | referred object)
- vis_unpaired  : logprobs for P(referring expression | negative object)
- lang_unpaired : logprobs for P(other expression | referred object)

Options:
- vis_rank: bool
- lang_rank: bool

Input can be one of the following four: 
1) logP(paired), of size (L, N, Mp1)
2) logP([paired, vis_unpaird]), of size (L, 2N, Mp1)
3) logP([paired, lang_unpaired]), of size (L, 2N, Mp1)
4) logP([paired, vis_unpaired, lang_unpaired]), of size (L, 3N, Mp1)

Output is a table of the follows:
- {logP([paired])}
- {logP([paired]), {logP([paired]), logP([vis_unpaired])} }  if vis_rank == true
- {logP([paired]), {logP([paired]), logP([lang_unpaired])} } if lang_rank == true
- {logP([paired]), {logP([paired]), logP([vis_unpaired])}, 
				   {logP([paired]), logP([lang_unpaired])} } if vis_rank == lang_rank == true

In order to feed into ParallelTable(), we need to prepare corresponding labels:
Assume seq is of long (D, N), where D = L-1
1) {pos_seq}
2) {pos_seq, {pos_seq, pos_seq}}
3) {pos_seq, {pos_seq, neg_seq}}
4) {pos_seq, {pos_seq, pos_seq}, {pos_seq, neg_seq}}

Note, each logP is of size (L, N, D), where L is the seq_length, N is batch_size and D is vocab_size
]]
local layer, parent = torch.class('nn.SplitGeneration', 'nn.Module')

function layer:__init(vis_rank, lang_rank)
	parent.__init(self)
	-- options
	assert(vis_rank~=nil and lang_rank~=nil)
	self.vis_rank = vis_rank
	self.lang_rank = lang_rank
end

function layer:updateOutput(input)

	local N = input:size(2)
	if self.vis_rank == false and self.lang_rank == false then
		-- input = logP(paired)
		self.output = {input}

	elseif self.vis_rank == true and self.lang_rank == false then
		-- input = logP([paired, vis_unpaired])
		local logP_paired = input[{ {}, {1,N/2}, {} }]
		local logP_visUnpaired = input[{ {}, {1+N/2, N}, {} }]
		self.output = {logP_paired, {logP_paired, logP_visUnpaired}}

	elseif self.vis_rank == false and self.lang_rank == true then
		-- input = logP([paired, lang_unpaired])
		local logP_paired = input[{ {}, {1, N/2}, {} }]
		local logP_langUnpaired = input[{ {}, {1+N/2, N}, {} }]
		self.output = {logP_paired, {logP_paired, logP_langUnpaired}}

	elseif self.vis_rank == true and self.lang_rank == true then
		-- input = logP([paired, vis_unpaired, lang_unpaired])
		local logP_paired = input[{ {}, {1,N/3}, {} }]
		local logP_visUnpaired = input[{ {}, {1+N/3, N*2/3}, {} }]
		local logP_langUnpaired = input[{ {}, {1+N*2/3, N}, {} }]
		self.output = {logP_paired, {logP_paired, logP_visUnpaired}, {logP_paired, logP_langUnpaired}}
	
	else
		error('No such option')
	end
	return self.output
end

function layer:updateGradInput(input, gradOutput)

	local N = input:size(2)
	if self.vis_rank == false and self.lang_rank == false then
		-- input = logP(paired)
		self.gradInput = gradOutput[1]

	elseif self.vis_rank == true and self.lang_rank == false then
		-- input = logP([paired, vis_unpaired])
		-- gradOutput = {dlogP_paired, {dlogP_paired, dlogP_visUnpaired}}
		self.gradInput:resizeAs(input):zero()
		local dlogP_paired = gradOutput[1] + gradOutput[2][1]
		local dlogP_visUnpaired = gradOutput[2][2]
		self.gradInput = torch.cat(dlogP_paired, dlogP_visUnpaired, 2)  -- along dim = 2

	elseif self.vis_rank == false and self.lang_rank == true then
		-- input = logP([paired, lang_unpaired])
		-- gradOutput = {dlogP_paired, {dlogP_paired, dlogP_langUnpaired}}
		self.gradInput:resizeAs(input):zero()
		local dlogP_paired = gradOutput[1] + gradOutput[2][1]
		local dlogP_langUnpaired = gradOutput[2][2]
		self.gradInput = torch.cat(dlogP_paired, dlogP_langUnpaired, 2)  -- along dim = 2

	elseif self.vis_rank == true and self.lang_rank == true then
		-- input = logP([paired, vis_unpaired, lang_unpaired])
		-- gradOutput = {dlogP_paired, {dlogP_paired, dlogP_visUnpaired}, {dlogP_paired, dlogP_langUnpaired}}
		self.gradInput:resizeAs(input):zero()
		local dlogP_paired = gradOutput[1] + gradOutput[2][1] + gradOutput[3][1]
		local dlogP_visUnpaired = gradOutput[2][2]
		local dlogP_langUnpaired = gradOutput[3][2]
		self.gradInput = torch.cat(torch.cat(dlogP_paired, dlogP_visUnpaired, 2), dlogP_langUnpaired, 2)

	else
		error('No such option')
	end
end









