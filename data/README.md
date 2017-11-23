This directory should contain the following data:
```
$COCO_PATH
├── detections
│   ├── refcoco_frcn.json
│   ├── refcoco_ssd.json
│   ├── refcoco+_frcn.json
│   ├── refcoco+_ssd.json
│   ├── refclef_edgebox.json
│   ├── refcocog_multibox.json
│   ├── refcocog_google_ssd.json
│   └── refcocog_umd_ssd.json
├── images
│   ├── mscoco
│   └── saiaprtc12
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
├── refcocog
│   ├── instances.json
│   └── refs(google).p
└── refclef
   	├── instances.json
	├── refs(unc).p
	└── refs(berkeley).p
```

Note, each detections/xxx.json contains 
``{'dets': ['box': {x, y, w, h}, 'image_id', 'object_id', 'score']}``. The ``object_id`` and ``score`` might be missing, depending on what proposal/detection technique we are using.
Download the detection/proposals from [here](http://bvisionweb1.cs.unc.edu/licheng/referit/data/detections.zip)
