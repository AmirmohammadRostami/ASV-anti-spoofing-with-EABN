# ASV-anti-spoofing-with-EABN

## Dependencies

1. Python and packages

    This code was tested on Python 3.8 with PyTorch 1.9.0.
    Other packages can be installed by:

    ```bash
    pip install -r requirements.txt
    ```

2. Kaldi-io-for-python

    kaldi-io-for-python is a python package that is used for reading and writing data of `ark,scp` kaldi format.
    See `README.md` in [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python) for installation.

3. MATLAB

   The LFCC feature adopted in this work is extracted via the MALTAB codes privided by ASVspoof2019 orgnizers.

## Dataset
   This work is conducted on [ASVspoof2019 Dataset](https://arxiv.org/pdf/1904.05441.pdf), which can be downloaded via https://datashare.ed.ac.uk/handle/10283/3336. It consists of two subsets, i.e. physical access (PA) for replay attacks and logical access (LA) for synthetic speech attacks.

## Start Your Project
   This repository mainly consists of two parts: (i) feature extraction and (ii) system training and evaluation.

### Feature extraction
   Three features are adopted in this repo, i.e. Spec, LFCC and CQT. The top script for feature extraction is `extract_feats.sh`, where the first step (Stage 0) is required to prepare dataset before feature extraction. It also provides feature extraction for Spec (Stage 1) and CQT (Stage 2), while for LFCC extraction, you need to run the `./baseline/write_feature_kaldi_PA_LFCC.sh` and `./baseline/write_feature_kaldi_LA_LFCC.sh` scripts. All features are required to be truncated by the Stage 4 in `extract_feats.sh`.

   Given your dataset directory in `extract_feats.sh`, you can run any stage (e.g. NUM) in the `extract_feats.sh` by
   ```bash
   ./extract_feats.sh --stage NUM
   ```
   For LFCC extraction, you need to run
   ```bash
   ./baseline/write_feature_kaldi_LA_LFCC.sh
   ./baseline/write_feature_kaldi_PA_LFCC.sh
   ```

### System training and evaluation
   This repo supports different system architectures, as configured in the `conf/training_mdl` directory. You can specify the system architecture, acoustic features in `start.sh`, then run the codes below to train and evaluate your models.
   ```bash
   ./start.sh
   ```
   Remember to rename your `runid` in `start.sh` to differentiate each configuration.


## Citation


## Contact
Feel free to contact us for any further information via below channels.

### Amirmohhammad Rostami:

- email: [*amirmohammadrostami@yahoo.com*](amirmohammadrostami@yahoo.com), [*a.m.rostami@aut.ac.ir*](a.m.rostami@aut.ac.ir)
- linkdin: [*amirmohammadrostami*](https://www.linkedin.com/in/amirmohammadrostami/)
- homepage: [*amirmohammadrostami*](https://ce.aut.ac.ir/~amirmohammadrostami/)
### Mohammad Mehdi Homayounpour
- email: [*homayoun@aut.ac.ir*](homayounaut.ac.ir)
- homepage: [*homayoun@aut.ac.ir*](https://aut.ac.ir/cv/2571/محمدمهدی-همایون-پور?slc_lang=fa&&cv=2571&mod=scv)
