# Minepy

Mutual information neural estimation

## About

This repository contains code implementation of information measures estimators.

1. Mutual information neural estimation (MINE):
   - Mine
   - Remine
2. Classification mutual information:
   - Classification mutual information
   - Classification conditional mutual information
     - Diff method
     - Generator method

## Install

```bash
git clone https://github.com/jrestrepo86/minepy.git
cd minepy/
pip install -e .
```

## Usage

1. [Mutual information neural estimation MINE module](minepy/mine/README.md)
2. [Classification mutual information](tests/test_class_mi/README.md)

## Bibliography

```
@inproceedings{belghazi2018mutual,
  title={Mutual information neural estimation},
  author={Belghazi, Mohamed Ishmael and Baratin, Aristide and Rajeshwar,
          Sai and Ozair, Sherjil and Bengio, Yoshua and Courville, Aaron and Hjelm, Devon},
  booktitle={International conference on machine learning},
  pages={531--540},
  year={2018},
  organization={PMLR}
}
inproceedings{choi2022combating, 
    title={Combating the instability of mutual information-based losses via regularization},
    author={Choi, Kwanghee and Lee, Siyeong}, 
    booktitle={Uncertainty in Artificial Intelligence},
    pages={411--421},
    year={2022},
    organization={PMLR}
}
@inproceedings{mukherjee2020ccmi,
    title={CCMI: Classifier based conditional mutual information estimation},
    author={Mukherjee, Sudipto and Asnani, Himanshu and Kannan, Sreeram},
    booktitle={Uncertainty in artificial intelligence},
    pages={1083--1093},
    year={2020},
    organization={PMLR}
}
@article{molavipour2021neural,
  title={Neural estimators for conditional mutual information using nearest neighbors sampling},
  author={Molavipour, Sina and Bassi, Germ{\'a}n and Skoglund, Mikael},
  journal={IEEE Transactions on Signal Processing},
  volume={69},
  pages={766--780},
  year={2021},
  publisher={IEEE}
}
```
