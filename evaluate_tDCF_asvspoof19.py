import sys
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt


def evaluate_(cm_score_file,asv_score_file):

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold, _,_ = em.compute_eer(tar_asv, non_asv)
    eer_cm, _ ,frr, far   = em.compute_eer(bona_cm, spoof_cm)

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)


    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    with open(cm_score_file.replace('.txt','_result.txt'), 'w') as f:
        print('ASV SYSTEM', file = f)
        print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100),file = f)
        print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100),file = f)
        print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100),file = f)
        print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100),file = f)

        print('\nCM SYSTEM',file = f)
        print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm * 100),file = f)

        print('\nTANDEM',file = f)
        print('   min-tDCF       = {:8.9f}'.format(min_tDCF),file = f)


    # Visualize ASV scores and CM scores
    plt.figure(figsize = (10,10))
    plt.subplots_adjust(hspace = 0.5, wspace = 0.2)

    ax = plt.subplot(221)
    plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
    plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
    plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
    plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
    plt.legend()
    plt.xlabel('ASV score')
    plt.ylabel('Density')
    plt.title('ASV score histogram')

    ax = plt.subplot(222)
    plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
    plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
    plt.legend()
    plt.xlabel('CM score')
    #plt.ylabel('Density')
    plt.title('CM score histogram')

    # Plot t-DCF as function of the CM threshold.
    # plt.figure()
    ax = plt.subplot(223)
    plt.plot(CM_thresholds, tDCF_curve)
    plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
    plt.xlabel('CM threshold index (operating point)')
    plt.ylabel('Norm t-DCF');
    plt.title('Normalized tandem t-DCF')
    plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
    plt.legend(('t-DCF', 'min t-DCF ({:.9f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
    plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
    plt.ylim([0, 1.5])

    ax = plt.subplot(224)
    frr*=100
    far*=100
    eer_cm *= 100
    plt.plot(frr, far)
    plt.title('FRR/FAR')
    plt.plot([0, 100], [0, 100])
    plt.scatter([eer_cm], [eer_cm], label='EER point')
    plt.xlabel('FRR')
    plt.ylabel('FAR')
    plt.xlim(-0.01, 100)
    plt.ylim(-0.01, 100)
    plt.text(eer_cm*1.3, eer_cm*1.3, f'EER = {eer_cm:.5f}')
    plt.legend()
    plt.savefig(cm_score_file.replace('.txt',f'_{eer_cm:.5f}.png'))
    plt.show()

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cm_score_file', action='store', type=str)
    parser.add_argument('--asv_score_file', action='store', type=str)
    args = parser.parse_args()

    evaluate_(args.cm_score_file,args.asv_score_file)