import os

pa_spec = {
        'train_scp': 'data/spec/PA_train/feats_slicing.scp',
        'train_utt2index': 'data/spec/PA_train/utt2index',
        'dev_scp': 'data/spec/PA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/spec/PA_dev/utt2index',
        'dev_utt2systemID': 'data/spec/PA_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/spec/PA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/spec/PA_eval/utt2index',
        'eval_utt2systemID': 'data/spec/PA_eval/utt2systemID',
        'eval_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.dev.gi.trl.scores.txt'
}

pa_lfcc = {
        'train_scp': 'data/lfcc/PA_train/feats_slicing.scp',
        'train_utt2index': 'data/lfcc/PA_train/utt2index',
        'dev_scp': 'data/lfcc/PA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/lfcc/PA_dev/utt2index',
        'dev_utt2systemID': 'data/lfcc/PA_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/lfcc/PA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/lfcc/PA_eval/utt2index',
        'eval_utt2systemID': 'data/lfcc/PA_eval/utt2systemID',
         'eval_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.dev.gi.trl.scores.txt'
}

pa_cqt = {
        'train_scp': 'data/cqt/PA_train/feats_slicing.scp',
        'train_utt2index': 'data/cqt/PA_train/utt2index',
        'dev_scp': 'data/cqt/PA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/cqt/PA_dev/utt2index',
        'dev_utt2systemID': 'data/cqt/PA_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/cqt/PA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/cqt/PA_eval/utt2index',
        'eval_utt2systemID': 'data/cqt/PA_eval/utt2systemID',
        'eval_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.dev.gi.trl.scores.txt'
}

la_spec = {
        'train_scp': 'data/spec/LA_train/feats_slicing.scp',
        'train_utt2index': 'data/spec/LA_train/utt2index',
        'dev_scp': 'data/spec/LA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/spec/LA_dev/utt2index',
        'dev_utt2systemID': 'data/spec/LA_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/spec/LA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/spec/LA_eval/utt2index',
        'eval_utt2systemID': 'data/spec/LA_eval/utt2systemID',
        'eval_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'
}

la_lfcc = {
        'train_scp': 'data/lfcc/LA_LFCC_train/feats_slicing.scp',
        'train_utt2index': 'data/lfcc/LA_LFCC_train/utt2index',
        'dev_scp': 'data/lfcc/LA_LFCC_dev/feats_slicing.scp',
        'dev_utt2index': 'data/lfcc/LA_LFCC_dev/utt2index',
        'dev_utt2systemID': 'data/lfcc/LA_LFCC_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/lfcc/LA_LFCC_eval/feats_slicing.scp',
        'eval_utt2index': 'data/lfcc/LA_LFCC_eval/utt2index',
        'eval_utt2systemID': 'data/lfcc/LA_LFCC_eval/utt2systemID',
        'eval_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'
}

la_cqt = {
        'train_scp': 'data/cqt/LA_train/feats_slicing.scp',
        'train_utt2index': 'data/cqt/LA_train/utt2index',
        'dev_scp': 'data/cqt/LA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/cqt/LA_dev/utt2index',
        'dev_utt2systemID': 'data/cqt/LA_dev/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/cqt/LA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/cqt/LA_eval/utt2index',
        'eval_utt2systemID': 'data/cqt/LA_eval/utt2systemID',
        'eval_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt',
        'dev_asv':'/opt/ssd/rostami/DS_10283_3336/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'
}

debug_feats = {
        'train_scp': 'data/debug_samples/feats_slicing.scp',
        'train_utt2index': 'data/debug_samples/utt2index',
        'dev_scp': 'data/debug_samples/feats_slicing.scp',
        'dev_utt2index': 'data/debug_samples/utt2index',
        'dev_utt2systemID': 'data/debug_samples/utt2systemID',
        'scoring_dir': 'scoring/cm_scores/',
        'eval_scp': 'data/debug_samples/feats_slicing.scp',
        'eval_utt2index': 'data/debug_samples/utt2index',
        'eval_utt2systemID': 'data/debug_samples/utt2systemID', 
}

data_prepare = {
        'pa_spec': pa_spec,
        'pa_cqt': pa_cqt,
        'pa_lfcc': pa_lfcc,
        'la_spec': la_spec,
        'la_cqt': la_cqt,
        'la_lfcc': la_lfcc,
        'debug_feats': debug_feats,
}

