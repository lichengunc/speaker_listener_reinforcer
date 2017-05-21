--[[
We will use torch to extract features.
This code is for extracting detected objects' feature.
Each input det = {det_id, 'box': [x1, y1, w, h], category_id, image_id, h5_id}

It sucks torch.hdf5 does not support too-large-size write. Its partial writing 
feature is missing and now DeepMind has dumped torch.
We have to save several hdf5's and use python's hdf5 binding (support partial write)
to combine them later. Or if you have a good machine with huge memory, you will 
never worry about out-of-memory nor partial writing.
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
-- extract det_feats
------------------------------------------------------------------------------------------
-- DataLoader
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5 = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}

-- also read dets.json
-- where dets = [{det_id, box, category_id, image_id, h5_id}]
local dets_json = 'cache/prepro/' .. opt.dataset .. '/dets.json'
local dets = utils.read_json(dets_json)['dets']

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
local feats = hdf5.open(feats_folder .. '/det_feats.h5', 'w')

-- extract det_feats
local Images = loader.Images
local det_feats = torch.zeros(#dets, 4096):float()

for bs = 1, #dets, opt.batch_size do
	local be = math.min(bs+opt.batch_size-1, #dets)
	-- current batch of dets
	local raw_dets = torch.zeros(be-bs+1, 3, 224, 224):cuda()
	local ib = 1
	for ix = bs, be do
		-- get h5-id
		local det = dets[ix]
		local h5_id = det['h5_id']
		assert(h5_id == ix, 'h5_id not match.')
		local img = Images[det['image_id']]
		-- get box
		local iw, ih = img['width'], img['height']
		local x1, y1, w, h = unpack(det['box'])
		local x2, y2 = math.max(x1+1, x1+w-1), math.max(y1+1, y1+h-1)
		-- raw img
		local file_name = img['file_name']
		local img_path = path.join(IMAGE_DIR, file_name)
		local raw_img = image.load(img_path) * 255  -- make range (1-255)
		-- protect range
		local nx1, ny1 = math.min(iw-1, math.max(1, x1)), math.min(ih-1, math.max(1, y1))
		local nx2, ny2 = math.max(2, math.min(iw, x2)), math.max(2, math.min(ih, y2))
		-- raw det, we maintain the aspect ratio and pad the margins with mean pixel value
		local raw_det = raw_img[{ {}, {ny1, ny2}, {nx1, nx2} }]
		-- feed into batch
		raw_dets[ib] = net_utils.prepro_ann(raw_det, true)  -- use prepro_ann
		ib = ib+1 
	end
	-- extract feats
	det_feats[{ {bs, be}, {} }] = cnn_fc7:forward(raw_dets):float()
	print(string.format('%s/%s done.', be, #dets))
end
feats:write('det_feats', det_feats:float())
print(string.format('det_feats.h5 extracted in %s', feats_folder))









