
# BadCLIP

Implementation of BadCLIP proposed in "BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP", accepted to CVPR2024.

```
@inproceedings{bai2023badclip,
  title={BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP},
  author={Bai, Jiawang and Gao, Kuofeng and Min, Shaobo and Xia, Shu-Tao and Li, Zhifeng and Liu, Wei},
  booktitle={CVPR},
  year={2024}
}
```

This code is based on [1, 2].

## Prepare Environment and Data
1. Please follow the instructions described in [1, 2] to install the environment, including dassl.

2. Please follow the instructions described in [2] to download the datasets and put them in /path/to/datasets.

3. Change paths in scripts/badclip/caltech101_train.sh and scripts/badclip/caltech101_test.sh, including data path, output path, and mode path.

## Backdoor Attack
Running the below command to perform the backdoor attack using the default settings on Caltech101.

	bash scripts/badclip/caltech101_train.sh

## Test on the Unseen Classes
Running the below command to test BadCLIP on unseen classes on Caltech101.
	
	bash scripts/badclip/caltech101_test.sh




[1] https://github.com/KaiyangZhou/Dassl.pytorch

[2] https://github.com/KaiyangZhou/CoOp
