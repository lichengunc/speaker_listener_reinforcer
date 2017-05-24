Torch implementation of CVPR 2017's referring expression paper "A Joint Speaker-Listener-Reinforcer Model for Referring Expressions"
==============================================
This repository contains code for both referring expression generation and referring expression comprehension task,
as described in this [paper](https://arxiv.org/abs/1612.09542).

Setup
====
* Clone the refer_baseline repository
```shell
# Make sure to clone with --recursive
git clone --recursive https://github.com/lichengunc/visdif_emb_guide2_reinforce.git
```
The ``recursive`` will help also clone the [refer API](https://github.com/lichengunc/refer) repo.
Then go to ``pyutils/refer`` and run ``make``.
* Download dataset and images, i.e., RefClef, RefCOCO, RefCOCO+, RefCOCOg from this [repo](https://github.com/lichengunc/refer.git), and save them into folder ``data/``.
* Download VGG-16-layer [model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md), and save both proto and prototxt into foloder ``models/vgg``.
* Download object proposals or object detections from [here](http://tlberg.cs.unc.edu/licheng/referit/data/detections.zip), and save the unzipped detections folder into data. We will use them for fully automatic comprehension task.

Preprocessing
====
We need to save the information of subset of MSCOCO for the use of referring expression.
By calling ``prepro.py``, we will save data.json and data.h5 into ``cache/prepro``
```Shell
python prepro.py --dataset refcoco --splitBy unc
```

Extract features
====
Before training or evaluation, we need to extract features.
* Extract region features for RefCOCO (UNC split):
```Shell
th scripts/extract_ann_feats.lua -dataset refcoco_unc -batch_size 50
```
* Extract image features for RefCOCO (UNC split):
```
th scripts/extract_img_feats.lua -dataset refcoco_unc -batch_size 50
```

Training
====
Make sure features are extracted - check if there exists ``ann_feats.h5`` and ``img_feats.h5`` in ``cache/feats/refcoco_unc/``, then run:
```shell
th train.lua 
```
We also provide options for training triplet loss.
There are two types of triplet loss:
* paired (ref_object, ref_expression) over unpaired (other_object, ref_expression), where other_object is mined from same image and perhaps same-category objects.
```shell
th train.lua -vis_rank_weight 1
```
* paired (ref_object, ref_expression) over unpaired (ref_object, other_expression).
```shell
th train.lua -lang_rank_weight 1
```
* Or you can train weith both triplet losses. Unfortunately, this seems working worse than "visual ranking" only for comprehension task.
```shell
th train.lua -vis_rank_weight 1 -lang_rank_weight 0.2
```

Evaluation
====
* Referring expression comprehension on ground-truth labled objects:
```shell
th eval_easy.lua -dataset refcoco_unc -split testA -mode 0 -id xxx
```
Note here ``mode = 0`` denotes evaluating using speaker model, ``model = 1`` denotes evaluating using listener model,
and finally ``model = 2'' denotes using the ensemble of speaker and listener models.

* Referring expression generation:
```shell
th eval_lang.lua -dataset refcoco_unc -split testA
```
Note, we have two testing splits, i.e., testA and testB for RefCOCO and RefCOCO+, if you are using UNC's split.

* Referring expressoin generation using unary and pairwise potentions.
```shell
th eval_lang.lua -dataset refcoco_unc -split testA -beam_size 10 -id xxx  # generate 10 sentences for each ref
th scripts/compute_beam_score.lua -id xx -split testA -dataset xxx  # compute cross (ref, sent) score
python eval_rerank.py --dataset xxx --split testA --model_id xxx --write_result 1  # call CPLEX to solve dynamic programming problem
```
You need to have IBM CPLEX installed in your machine. First run ``eval_lang.lua`` with beam_size 10, then call ``compute_beam_score.lua`` to compute cross (ref, sent) scores, finally call ``eval_rerank.py`` to pick the sentences with highest score.
For more details, check ``eval_rerank.py``


Fully automatic comprehension using detection/proposal
====
We provide proposals/detections for each dataset. Please follow the setup section to download the ``detections.zip`` and unzip it into ``data`` folder.
Same as above, we need to do pre-processing and feature extraction for all the detected regions.
* Run ``prepro_dets.py`` to add ``det_id`` and ``h5_id`` for each detected region, which will save ``dets.json`` into ``cache/prepro/refcoco_unc`` folder.
```shell
python scripts/prepro_dets.py --dataset refcoco --splitBy unc --source data/detections/refcoco_ssd.json
python scripts/prepro_dets.py --dataset refcoco+ --splitBy unc --source data/detections/refcoco+_ssd.json
python scripts/prepro_dets.py --dataset refcocog --splitBy google --source data/detections/refcocog_google_ssd.json
``` 
* Extract features for each region:
```shell
th scripts/extract_det_feats.lua -dataset refcoco_unc
```
* Then we can evaluate the comprehension accuracies on detected objects/proposals:
```shell
th eval_dets.lua -dataset refcoco_unc -split testA -id xxx
```

Pretrained models on RefCOCO (UNC)
====
We provided two pretrained models [here](http://tlberg.cs.unc.edu/licheng/referit/visdif_emb_guide2_reinforce/models/refcoco_unc.zip). Specifically they are trained using
* no_rank: ``th train.lua -id no_rank -vis_rank_weight 0 -lang_rank_weight 0``
* 0: ``th train.lua -id 0 -vis_rank_weight 1 -lang_rank_weight 0.1``

<table>
<tr><th> Ground-truth Box </th><th> Detected Regions(ssd) </th></tr>
<tr><td>

| System | testA | testB | 
|:-------|:-----:|:-------:|
| no_rank (speaker) | 71.10\% | 74.01\% |
| no_rank (listener) | 76.91\% | 80.10\% |
| no_rank (ensemble 0.2) | 78.01\% | 80.65\% |
| 0 (speaker)  | 78.95\% | 80.22\% |
| 0 (listener) | 77.97\% | 79.86\% |
| 0 (ensemble 0.2) | 80.08\% | 81.73\% |
</td><td>

| model | test A | test B |
|--|--|--|
| no_rank (speaker) | 69.15\% | 61.96\% |
| no_rank (listener) | 72.65\% | 62.69\% |
| no_rank (ensemble 0.2) | 72.78\% | 64.38\% |
| 0 (speaker)  | 72.88\% | 63.43\% |
| 0 (listener) | 72.94\% | 62.98\% |
| 0 (ensemble 0.2) | 73.78\% | 63.83\% |
</td></tr> </table>

Pretrained models on RefCOCO+ (UNC)
====
We provided two pretrained models [here](http://tlberg.cs.unc.edu/licheng/referit/visdif_emb_guide2_reinforce/models/refcoco+_unc.zip). Specifically they are trained using
* no_rank: ``th train.lua -dataset refcoco+_unc -id no_rank -vis_rank_weight 0 -lang_rank_weight 0``
* 0: ``th train.lua -dataset refcoco+_unc -id 0 -vis_rank_weight 1 -lang_rank_weight 0``

<table>
<tr><th> Ground-truth Box </th><th> Detected Regions(ssd) </th></tr>
<tr><td>

| model | test A | test B |
|--|--|--|
| no_rank (speaker) | 57.46\% | 53.71\% |
| no_rank (listener) | 63.34\% | 58.42\% |
| no_rank (ensemble 0.3) | 64.02\% | 59.19\% |
| 0 (speaker)  | 64.60\% | 59.62\% |
| 0 (listener) | 63.10\% | 58.19\% |
| 0 (ensemble 0.3) | 65.40\% | 60.73\% |
</td><td>

| model | test A | test B |
|--|--|--|
| no_rank (speaker) | 55.97\% | 46.45\% |
| no_rank (listener) | 58.68\% | 48.23\% |
| no_rank (ensemble 0.3) | 59.80\% | 49.34\% |
| 0 (speaker)  | 60.43\% | 48.74\% |
| 0 (listener) | 58.68\% | 47.68\% |
| 0 (ensemble 0.3) | 60.48\% | 49.36\% |
</td></tr> </table>

Pretrained models on RefCOCOg (Google)
====
We provided two pretrained models [here](http://tlberg.cs.unc.edu/licheng/referit/visdif_emb_guide2_reinforce/models/refcocog_google.zip). Specifically they are trained using
* no_rank: ``th train.lua -dataset refcocog_google -id no_rank -vis_rank_weight 0 -lang_rank_weight 0``
* 0.2: ``th train.lua -dataset refcoco+_unc -id 0.2 -vis_rank_weight 1 -lang_rank_weight 0.5``
* 0.4 (branch: refcocog2): ``th train.lua -dataset refcoco+_unc -id 0.4 -vis_rank_weight 1 -lang_rank_weight 1``

<table>
<tr><th> Ground-truth Box </th><th> Detected Regions(ssd) </th></tr>
<tr><td>

| model | val | 
|:-------|:-----:|
| no_rank (speaker) | 64.07\% | 
| no_rank (listener) | 71.72\% | 
| no_rank (ensemble 0.2) | 72.43\% | 
| 0.2 (speaker)  | 72.63\% | 
| 0.2 (listener) | 72.02\% | 
| 0.2 (ensemble 0.2) | 74.19\% | 
</td><td>

| model | val | 
|:-------|:-----:|
| no_rank (speaker) | 57.03\% | 
| no_rank (listener) | 58.32\% | 
| no_rank (ensemble 0.2) | 60.46\% | 
| 0.4 (speaker)  | 59.51\% | 
| 0.4 (listener) | 57.72\% | 
| 0.4 (ensemble 0.2) | 59.84\% | 
</td></tr></table>

TODO
====
* automatic referring expression generation given just an image
Current code only support referring expression geneneration given ground-truth bounding-box (with label). It shouldn't take too much effort to finish this. Just add some more functions inside DetsLoader.lua


Conclusion
====
The joint training of speaker and listener is helpful to each other.
Though this is not reflected in this repo as we only show the joint training results, we refer readers to check our CVPR2017 [paper](https://arxiv.org/abs/1612.09542) to check this improvement over single speaker or single listener.
Generally, ``+MMI'' gives additional boost on speaker, but this has no effect on listener.
Overall the reinforcer has some effect but less than incorporating speaker.
The joint training of all three modules make the best results so far.

The difference between master and refcocog2 is we forbid sampling different-type objects (by stopping their gradient backpropagation) in refcocog2. The sampling of different-type objects is good for ground-truth comprehension for all three datasets, but make awkward performance on refcocog's detections. To be investigated...

Future work involves larger dataset collection.












