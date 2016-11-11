require 'hdf5'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.modules.SplitEmbedding'
require 'misc.modules.SplitGeneration'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local make_data = require 'misc.make_data'
torch.manualSeed(8)

-- option
local opt = {}
opt.dataset = 'refcoco_unc'
opt.id = '0'
opt.gpuid = 0
opt.seq_per_ref = 3
opt.lang_encoding_size = 512

-- load model checkpoint
local model_path = path.join('models', opt.dataset, 'model_id' .. opt.id .. '.t7')
local checkpoint = torch.load(model_path)
local protos = checkpoint.protos

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'use_context', 'use_ann', 'use_location', 'margin', 'dif_ann', 'dif_location', 'dif_num',
			   'dif_source', 'dif_pool'}
for k, v in pairs(fetch) do opt[v] = checkpoint.opt[v] end

-- Create the data loader
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5   = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_json = data_json, data_h5 = data_h5}
-- also load extracted features: call scripts/extract_xxx_feats before training!
local feats_dir = 'cache/feats/' .. opt.dataset
local featsOpt = {  ann = feats_dir .. '/ann_feats.h5',
					img = feats_dir .. '/img_feats.h5',
					det = feats_dir .. '/det_feats.h5',
					window2 = feats_dir .. '/window2_feats.h5',
					window3 = feats_dir .. '/window3_feats.h5',
					window4 = feats_dir .. '/window4_feats.h5',
					window5 = feats_dir .. '/window5_feats.h5' }
loader:loadFeats(featsOpt)

-- run make_data
make_data.make_new_data(protos, loader, 'train', 50000, opt)









