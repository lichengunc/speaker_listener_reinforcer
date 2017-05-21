--[[
We will use torch to extract features.
This code is used to extract image features.
]]
require 'misc.DataLoader'
require 'torch'
require 'loadcaffe'
require 'nn'
-- local
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
-- cuda
require 'cutorch'
require 'cunn'

------------------------------------------------------------------------------------------
-- prepare data
------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Extract features')
cmd:text()
cmd:text('Options')
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset_splitBy')
cmd:option('-cnn_proto', 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to cnn prototxt')
cmd:option('-cnn_model', 'models/vgg/VGG_ILSVRC_16_layers.caffemodel','path to cnn model')
cmd:option('-batch_size', 50, 'batch of images to be fed into cnn')
cmd:option('-gpuid', 0, 'gpu_id to be used')
cmd:option('-window_scale', 2, 'scale factor for window size, which is centered by each instance')
cmd:option('-view', 0, 'set to 1 if you have torch-opencv installed and want to check image.')
local opt = cmd:parse(arg)

------------------------------------------------------------------------------------------
-- load cnn model
------------------------------------------------------------------------------------------
-- set gpu
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
-- set cudnn to be fastest, I don't like to save memory
local cudnn = require 'cudnn'
cudnn.fastest, cudnn.benchmark = true, true
-- load truncated cnn
local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
local cnn_fc7 = net_utils.build_cnn(cnn_raw)
cnn_fc7:evaluate()
cnn_fc7:cuda()

------------------------------------------------------------------------------------------
-- extract window_feats
------------------------------------------------------------------------------------------
-- DataLoader
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5 = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}

-- Image Directory
local IMAGE_DIR
if string.match(opt.dataset, 'coco') then
	IMAGE_DIR = 'data/images/mscoco/images/train2014'
elseif string.match(opt.dataset, 'clef') then
	IMAGE_DIR = 'data/images/saiapr_tc-12'
else
	print('No image directory prepared for ' .. opt.dataset)
	os.exit()
end

-- Prepare h5 file
local feats_folder = 'cache/feats/' .. opt.dataset
if not utils.file_exists('cache/feats') then os.execute('mkdir cache/feats') end
if not utils.file_exists(feats_folder) then os.execute('mkdir ' .. feats_folder) end
local feats = hdf5.open(feats_folder .. '/window' .. opt.window_scale .. '_feats.h5', 'w')

-- extract
local Images = loader.Images
local anns = loader.anns
local window_feats = torch.zeros(#anns, 4096):float()
local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(3,1,1)

for bs = 1, #anns, opt.batch_size do
	local be = math.min(bs+opt.batch_size-1, #anns)

	local raw_window_anns = torch.zeros(be-bs+1, 3, 224, 224):cuda()
	local ib = 1
	for ix = bs, be do

		-- get h5_id
		local ann = anns[ix]
		local h5_id = ann['h5_id']
		assert(h5_id == ix, 'h5_id not match.')
		local img = Images[ann['image_id']]
		-- get box
		local iw, ih = img['width'], img['height']
		local x1, y1, w, h = unpack(ann['box'])
		local x2, y2 = math.max(x1+1, x1+w-1), math.max(y1+1, y1+h-1)
		-- raw img
		local file_name = img['file_name']
		local img_path = path.join(IMAGE_DIR, file_name)
		local raw_img = image.load(img_path) * 255  -- make range (1-255)
		if raw_img:size(1) == 1 then raw_img = raw_img:expand(3, raw_img:size(2), raw_img:size(3)) end
		-- window range
		local scale = opt.window_scale
		local cx, cy = math.floor(x1+w/2), math.floor(y1+h/2)
		local B = math.floor(math.max(w, h)*scale)
		local nx1, nx2, ny1, ny2 = math.floor(cx-B/2), math.floor(cx+B/2-1), math.floor(cy-B/2), math.floor(cy+B/2-1)
		nx1, nx2, ny1, ny2 = math.max(1, nx1), math.min(iw, nx2), math.max(1, ny1), math.min(ih, ny2)
		-- make window_ann
		local shift_x, shift_y = 1-math.floor(cx-B/2), 1-math.floor(cy-B/2)
		local wnx1, wnx2, wny1, wny2 = nx1+shift_x, nx2+shift_x, ny1+shift_y, ny2+shift_y
		local window_ann = torch.zeros(raw_img:size(1), B, B)
		window_ann:add(vgg_mean:typeAs(window_ann):expandAs(window_ann))  -- pad with mean value
		window_ann[{ {}, {wny1, wny2}, {wnx1, wnx2} }] = raw_img[{ {}, {ny1, ny2}, {nx1, nx2} }]
		-- view window_ann
		if opt.view == 1 then
			local raw_ann = raw_img[{ {}, {math.max(y1, 1), math.min(ih, math.max(y1+h-1, y1+1))}, 
			{math.max(x1, 1), math.min(iw, math.max(x1+w-1, x1+1))} }]
			raw_ann = image.scale(raw_ann, B, B)
			utils.viewRawImg(torch.cat(window_ann, raw_ann, 3))
		end
		-- feed into batch
		raw_window_anns[ib] = net_utils.prepro_img(window_ann, true)
		ib = ib+1
	end

	-- extract feats
	window_feats[{ {bs, be}, {} }] = cnn_fc7:forward(raw_window_anns):float()
	print(string.format('%s/%s done.', be, #anns))
end
feats:write('window_feats', window_feats:float()) 
print(string.format('window_feats.h5 extracted in %s', feats_folder))















