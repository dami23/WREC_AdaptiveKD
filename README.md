This project is the pytorch implementation for "Adaptive Distilling Target- and Interaction-Aware Knowledge for Weakly Supervised Referring Expression Comprehension". 


### Preliminary
1. Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install mask-faster-rcnn, REFER and refer-parser2. Follow Step 1 & 2 in Training to prepare the data and features.

2. Please follow the step in [DTWREG](https://github.com/insomnia94/DTWREG) to acquire the parsed discriminative triads.

The experiments are conducted on one GPU (NVIDIA RTX 3090ti).

- python == 3.7.13
- pytorch == 1.10
### Feature Encoding
1. follow the feature extraction in MattNet

2. extract ann_pool5 and ann_fc7 feats using py27 + pytorch 0.4.1

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/extract_mrcn_ann_fc7_feats.py --dataset {DATASET} --splitBy {SPLITBY}

### Training and evaluation
1. train the teacher model

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/train_teacher.py --dataset {DATASET} --splitBy {SPLITBY} --exp_id {EXP_ID}

2. train the student model

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/train_student.py --dataset {DATASET} --splitBy {SPLITBY} --exp_id {EXP_ID}

3. evaluate the models,

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/eval.py --dataset {DATASET} --splitBy {SPLITBY} --split {SPLIT} --id {EXP_ID}

   {DATASET} = refcoco, refcoco+, refcocog. {SPLITBY} = unc for refcoco and refcoco+, google for refcocog.


### Pretrained Models
The trained models on three benchmarks can be downloaded [here](https://drive.google.com/drive/folders/183BmPhVlt8NYfZdWq5LGYB5XAG6ohI0S?usp=share_link).

### Acknowledgement
The code is based on [DTWREG](https://github.com/insomnia94/DTWREG/).
