import numpy as np
import os
import random
import shutil
import time
import warnings
from collections import defaultdict
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def prediction(val_loader, model, device, output_file, utt2systemID_file):
    
    # switch to evaluate mode
    utt2scores = defaultdict(list) 
    model.eval()
    # find = False
    # samples = 0
    # print(len(val_loader))
    with torch.no_grad():
        for i, (utt_list, input, target) in enumerate(val_loader):
            if i % 20 == 0:
                print(i,i/len(val_loader))
            input  = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))
            # compute output
            (_,output),_,_ = model(input)
            score = output[:,0] # use log-probability of the bonafide class for scoring 

            for index, utt_id in enumerate(utt_list):
                curr_utt = ''.join(utt_id.split('-')[0])
                utt2scores[curr_utt].append(score[index].item()) 
   
        # first do averaging
        with open(utt2systemID_file, 'r') as f:
            temp = f.readlines()
        content  = [x.strip() for x in temp]
        utt_list = [x.split()[0] for x in content]
        id_list  = [x.split()[1] for x in content]

        eerfile = output_file+'.eer'
        with open(output_file, 'w') as f, open(eerfile, 'w') as eerf:
            for index, utt_id in enumerate(utt_list):
                score_list = utt2scores[utt_id]
                assert score_list != [], '%s' %utt_id   
                avg_score  = reduce(lambda x, y: x + y, score_list) / len(score_list)
                spoof_id = id_list[index]
                if spoof_id == '-':
                    f.write('%s %s %s %f\n' % (utt_id, '-', 'bonafide', avg_score))
                    eerf.write('%f target\n' %avg_score)
                else: 
                    f.write('%s %s %s %f\n' % (utt_id, spoof_id, 'spoof', avg_score))
                    eerf.write('%f nontarget\n' %avg_score)

def visualize(val_loader, model, device, output_file, centers):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    centers = centers['btc_loss.centers'].cpu().numpy()
    embeded_vectores = [centers[0], centers[1]]
    embeded_vectores_lbl = [-1, -1]
    N = len(val_loader)
    lbls = {}
    count_lbls = []
    maps = []
    uttidx = {}
    uttn = {}
    utt2vectores = defaultdict(list)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for j, (utt_list, input, target) in enumerate(val_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))
            # compute output
            (feats, _), _, attention_maps = model(input)
            input = input.cpu().numpy()
            feats = feats.cpu().numpy()
            attention_maps = attention_maps.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(len(feats)):
                utt = utt_list[i]
                if j < 15:
                    # inp
                    plt.figure(figsize=(4, 3))
                    plt.imshow(input[i][0],
                               cmap='gray_r')
                    plt.title(id2lbl[target[i]])
                    plt.tight_layout()
                    plt.axis('off')
                    plt.savefig(
                        f'{output_file}_{id2lbl[target[i]]}_{i}_{j}inputs.png')
                    plt.close()
                    # _maps
                    plt.figure(figsize=(4, 3))
                    plt.imshow(attention_maps[i][0],
                               cmap='gray_r')
                    plt.title(id2lbl[target[i]])
                    plt.tight_layout()
                    plt.axis('off')
                    plt.savefig(
                        f'{output_file}_{id2lbl[target[i]]}_{i}_{j}_maps.png')
                    plt.close()
                # print(f'{output_file}_{id2lbl[lbl]}_maps.png')
                try:
                    idx, n = uttidx[utt], uttn[utt]
                    n += 1
                    embeded_vectores_lbl[idx] = embeded_vectores_lbl[idx] + \
                        ((feats[i]-embeded_vectores_lbl[idx])/n)
                    uttn[utt] = n
                except:
                    (idx, n) = len(embeded_vectores_lbl), 1
                    embeded_vectores_lbl.append(target[i])
                    embeded_vectores.append(feats[i])
                    uttn[utt] = n
                    uttidx[utt] = idx
                try:
                    idx = lbls[target[i]]
                    count_lbls[idx] += 1
                    maps[i] = maps[i] + \
                        ((attention_maps[i]-maps[i])/count_lbls[idx])
                except:
                    lbls[target[i]] = len(count_lbls)
                    count_lbls.append(1)
                    maps.append(attention_maps[i])
            if j == 100:
                break

    embeded_vectores = np.array(embeded_vectores)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeded_vectores)
    X_tsne = TSNE(n_components=2, learning_rate=50,
                  n_iter=500, ).fit_transform(embeded_vectores)
    uniq_lbls = list(set(embeded_vectores_lbl))
    plt.figure(figsize=(5, 5))
    markercolor = [('.', 'b'),
                   ('.', 'c'),
                   ('.', 'm'),
                   ('.', 'y'),
                   ('8', 'b'),
                   ('8', 'c'),
                   ('8', 'm'),
                   ('8', 'y'),
                   ('+', 'b'),
                   ('+', 'c'),
                   ('+', 'm'),
                   ('+', 'y'),
                   ('p', 'b'),
                   ('p', 'c'),
                   ('p', 'm'),
                   ('p', 'y'),
                   ('v', 'b'),
                   ('v', 'c'),
                   ('v', 'm'),
                   ('v', 'y')]
    # PCA
    for ii, lbl in enumerate(uniq_lbls):
        label = lbl
        if label == -1:
            continue
        idxs = np.where(embeded_vectores_lbl == lbl)
        x = X_pca[idxs]
        plt.scatter(x[:, 0], x[:, 1], label=id2lbl[lbl], marker=markercolor[ii][0] if lbl !=
                    0 else '*', s=20, c=markercolor[ii][1] if lbl != 0 else 'g', alpha=0.25 if lbl != 0 else 0.75)
    plt.scatter(X_pca[:2, 0], X_pca[:2, 1],
                label='centers', marker='x', s=100, c='k')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'{output_file}_2D_features_pca.png')
    print(f'{output_file}_2D_features_pca.png')
    plt.cla()
    # TSNE
    for ii, lbl in enumerate(uniq_lbls):
        label = lbl
        if label == -1:
            continue
        idxs = np.where(embeded_vectores_lbl == lbl)
        x = X_tsne[idxs]
        plt.scatter(x[:, 0], x[:, 1], label=id2lbl[lbl], marker=markercolor[ii][0] if lbl !=
                    0 else '*', s=20, c=markercolor[ii][1] if lbl != 0 else 'g', alpha=0.25 if lbl != 0 else 0.75)
    plt.scatter(X_tsne[:2, 0], X_tsne[:2, 1],
                label='centers', marker='x', s=100, c='k')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'{output_file}_2D_features_tsne.png')
    plt.cla()
    print(f'{output_file}_2D_features_tsne.png')
    for ii, lbl in enumerate(uniq_lbls):
        plt.figure(figsize=(4, 3))
        plt.imshow(maps[ii][0],
                   cmap='gray_r')
        plt.title(id2lbl[lbl])
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'{output_file}_{id2lbl[lbl]}_maps.png')
        print(f'{output_file}_{id2lbl[lbl]}_maps.png')