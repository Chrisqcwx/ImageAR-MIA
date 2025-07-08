# ICAS: Detecting Training Data from Autoregressive Image Generative Models

ACM MM 2025

[arxiv](https://arxiv.org/abs/2507.05068)

## Usage

Get the VAR repository:
```shell
git clone git@github.com:FoundationVision/VAR.git
rm ./VAR/README.md
mv ./VAR/* .
rmdir ./VAR
```

Modify the checkpoint and imagenet folder in line 6-7 and run membership inference:
```shell
# run baseline
python baseline.py
# run ICAS
python icas.py
```


## Citation
```
@misc{yu2025icas,
      title={ICAS: Detecting Training Data from Autoregressive Image Generative Models}, 
      author={Hongyao Yu and Yixiang Qiu and Yiheng Yang and Hao Fang and Tianqu Zhuang and Jiaxin Hong and Bin Chen and Hao Wu and Shu-Tao Xia},
      year={2025},
      eprint={2507.05068},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05068}, 
}
```
