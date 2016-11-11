require 'nn'
--[[
Denotation:
- sc_paired       : score for P(referring expression, referred object)
- sc_vis_unpaired : score for P(referring expression, negative object)
- sc_lang_unpaired: score for P(negative expression, referred object)

Options:
- vis_rank : bool
- lang_rank: bool

Input can be one of the following three:
1) [sc_paired, sc_vis_unpaired] : (2n, )
2) [sc_paired, sc_lang_unpaired]: (2n, )
3) [sc_paired, sc_vis_unpaired, sc_lang_unpaired]: (3n, )

Output a table of {[positive score, negative score], [], ...} 
so that we can feed into ParallelCriterion() and MarginRankingCriterion()
1) {[sc_paired, sc_vis_unpaired]} if vis_rank = true
2) {[sc_paired, sc_lang_unpaired]} if lang_rank = true
3) {[sc_paired, sc_vis_unpaired], [sc_paired, sc_lang_unpaired]} if vis_rank == lang_rank == true
]]
local layer, parent = torch.class('nn.SplitEmbedding', 'nn.Module')

function layer:__init(vis_rank, lang_rank)
	parent.__init(self)
	-- options
	assert(vis_rank~=nil and lang_rank~=nil)
	self.vis_rank = vis_rank
	self.lang_rank = lang_rank
	assert(vis_rank == true or lang_rank == true, 
		'at least one of vis_rank and lang_rank should be true.')
end

function layer:updateOutput(input)
	assert(input:nDimension() == 1)
	local N = input:nElement()
	if self.vis_rank == true and self.lang_rank == false then
		-- input = [paired, vis_unpaired]
		local paired = input[{{1, N/2}}]
		local vis_unpaired = input[{ {1+N/2, N} }]
		self.output = {{paired, vis_unpaired}}

	elseif self.vis_rank == false and self.lang_rank == true then
		-- input = [paired, lang_unpaired]
		local paired = input[{{1, N/2}}]
		local lang_unpaired = input[{ {1+N/2, N} }]
		self.output = {{paired, lang_unpaired}}

	else
		-- input = [paired, vis_unpaired, lang_unpaired]
		local paired = input[{{1, N/3}}]
		local vis_unpaired = input[{ {1+N/3, N*2/3} }]
		local lang_unpaired = input[{ {1+N*2/3, N} }]
		self.output = {{paired, vis_unpaired}, {paired, lang_unpaired}}
	end
	-- return
	return self.output
end

function layer:updateGradInput(input, gradOutput)

	local N = input:nElement()
	self.gradInput:resizeAs(input):zero()
	if self.vis_rank == true and self.lang_rank == false then
		-- input = [paired, vis_unpaired]
		local dpaired = gradOutput[1][1]
		local dvis_unpaired = gradOutput[1][2]
		self.gradInput = torch.cat(dpaired, dvis_unpaired)  
	elseif self.vis_rank == false and self.lang_rank == true then
		-- input = [paired, lang_unpaired]
		local dpaired = gradOutput[1][1]
		local dlang_unpaired = gradOutput[1][2]
		self.gradInput = torch.cat(dpaired, dlang_unpaired)  
	else
		-- input = [paired, vis_unpaired, lang_unpaired]
		local dpaired = gradOutput[1][1] + gradOutput[2][1]
		local dvis_unpaired = gradOutput[1][2]
		local dlang_unpaired = gradOutput[2][2]
		self.gradInput = torch.cat(torch.cat(dpaired, dvis_unpaired) , dlang_unpaired)
	end
	-- return 
	return self.gradInput
end


























