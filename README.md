# ASV-anti-spoofing-with-EABN
Efficient Attention Branch Network with Combined Loss Function for Automatic Speaker Verification Spoof Detection
Many endeavors have sought to develop countermeasure techniques as enhancements on Automatic Speaker Verification (ASV) systems, in order to make them more robust against spoof attacks. As evidenced by the latest ASVspoof 2019 countermeasure challenge, models currently deployed for the task of ASV are, at their best, devoid of suitable degrees of generalization to unseen attacks. Upon further investigation of the proposed methods, it appears that a broader three-tiered viewof the proposed systems; comprised of the classifier, feature extraction phase, and model loss function, may to some extent lessen the problem. Accordingly, the present study proposes the Efficient Attention Branch Network (EABN) modular architecture with a combined loss function to address the generalization problem. The EABN architecture is based on attention and perception branches; the purpose of the attention branch—also interpretable from a human’s point of view—is to produce an attention mask meant to improve classification performance. The perception branch, on the other hand, is used for the primary purpose of the problem at hand, that is, spoof detection. The new EfficientNet-A0 ([paper](https://arxiv.org/abs/2012.15695)/[code](https://github.com/AmirmohammadRostami/KeywordsSpotting-EfficientNet-A0)) architecture was employed for the perception branch, with nearly ten times fewer parameters and approximately seven times fewer floating-point operations than the top performing SE-Res2Net50 network. The final evaluation results on ASVspoof 2019 dataset suggest an EER = 0.86% and t-DCF = 0.0239 in the Physical Access (PA) scenario using the log-PowSpec input feature, the EfficientNet-A0 for the perceptionbranch, and the combined loss function. Furthermore, using the LFCC input feature, the SE-Res2Net50 for the perception branch, and the combined loss function, the proposed modelperformed at figures of EER = 1.89% and t-DCF = 0.507 in the Logical Access (LA) scenario, which to the best of our knowledge, is the best single system ASV spoofing countermeasure.

More details of architecture, experiments, and results can be found in our [published](https://link.springer.com/article/10.1007/s00034-023-02314-5) paper.

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
- email: [*homayoun@aut.ac.ir*](homayounaut@aut.ac.ir)
- homepage: [*homayounpour*](https://aut.ac.ir/cv/2571/محمدمهدی-همایون-پور?slc_lang=fa&&cv=2571&mod=scv)
### Mohammad Mehdi Homayounpour
- email: [*nickabadi@aut.ac.ir*](nickabadi@aut.ac.ir)
- homepage: [*nickabadi*](https://aut.ac.ir/cv/2387/%d8%a7%d8%ad%d9%85%d8%af%20%d9%86%db%8c%da%a9%20%d8%a2%d8%a8%d8%a7%d8%af%db%8c)
