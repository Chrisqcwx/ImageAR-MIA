# ImageAR-MIA
ACM MM 2025: ICAS: Detecting Training Data from Autoregressive Image Generative Models

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



