#!/bin/bash
# This script extracts Spec, CQT, LFCC features for:
# ASVspoof2019 PA train, PA dev, PA eval, LA train, LA dev, LA eval

. ./cmd.sh
. ./path.sh
set -e
specdir=`pwd`/raw_feats/spec
datadir=/home/rostami/DS_10283_3336/

stage=3


. ./parse_options.sh    

mkdir -p data/PA_train data/PA_dev data/PA_eval data/LA_train data/LA_dev data/LA_eval    

if [ $stage -le 0 ]; then
   echo "Stage 0: prepare dataset."
   for access_type in PA LA;do
       protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.train.trn.txt
       awk '{print $2" "$4}' $protofile >data/${access_type}_train/utt2systemID    
       awk '{print $2" "$1}' $protofile >data/${access_type}_train/utt2spk    
       feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_train/utt2spk >data/${access_type}_train/spk2utt    
       awk -v dir="${datadir}/${access_type}" -v type="${access_type}_train" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_train/wav.scp    
       
       for dataset in dev eval;do
           protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.${dataset}.trl.txt
           awk '{print $2" "$4}' $protofile >data/${access_type}_${dataset}/utt2systemID    
           awk '{print $2" "$1}' $protofile >data/${access_type}_${dataset}/utt2spk    
           feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_${dataset}/utt2spk >data/${access_type}_${dataset}/spk2utt    
           awk -v dir="${datadir}/${access_type}" -v type="${access_type}_${dataset}" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_${dataset}/wav.scp    
       done
    done
    echo "dataset finished"
     
fi

if [ $stage -le 1 ]; then
   echo "Stage 1: extract Spec feats."
   mkdir -p data/spec    
   for name in PA_train PA_dev PA_eval LA_train LA_dev LA_eval; do
       [ -d data/spec/${name} ]   cp -r data/${name} data/spec/${name}    
       feats_extraction/make_spectrogram.sh --feats-config conf/feats/spec.conf --nj 80 --cmd "$train_cmd" \
             data/spec/${name} exp/make_spec $specdir    
   done
   echo "Spec feats done"
    
fi

if [ $stage -le 2 ]; then
   echo "Stage 2: extract CQT feats."
   python3 -W ignore feats_extraction/compute_CQT.py --out_dir data/cqt --access_type PA --param_json_path conf/feats/cqt_48bpo_fmin15.json --num_workers 60
#    python3 feats_extraction/compute_CQT.py --out_dir data/cqt --access_type LA --param_json_path conf/feats/cqt_48bpo_fmin15.json --num_workers 60    
   python3 -W ignore feats_extraction/GenLPCQTFeats_kald
   i.py --access_type PA --work_dir data/cqt
#    python3 feats_extraction/GenLPCQTFeats_kaldi.py --access_type LA --work_dir data/cqt    
    echo "DONE stage 2"
   # uncomment below for removing numpy data to save space.
   #for name in PA_train PA_dev PA_eval LA_train LA_dev LA_eval; do
   #    rm -rf data/cqt/${name}/*.npy    
   #done
   # find data/cqt/ -type f -name "*.npy" -exec rm {} \;
    
fi

if [ $stage -le 3 ]; then
   echo "Stage 3: truncate features and generate labels."
   for name in PA_train PA_dev; do
       for feat_type in cqt; do
           echo "Processing $name $feat_type"
           python3 feats_extraction/feat_slicing.py --in-scp data/${feat_type}/${name}/feats.scp --out-scp data/${feat_type}/${name}/feats_slicing.scp --out-ark data/${feat_type}/${name}/feats_slicing.ark    
           python3 feats_extraction/convertID2index.py --scp-file data/${feat_type}/${name}/feats_slicing.scp --sysID-file data/${name}/utt2systemID --out-file data/${name}/utt2index --access-type ${name:0:2}    
       done
   done
    
fi


# #!/bin/bash
# # This script extracts Spec, CQT, LFCC features for:
# # ASVspoof2019 PA train, PA dev, PA eval, LA train, LA dev, LA eval

# . ./cmd.sh
# . ./path.sh
# set -e
# specdir=`pwd`/raw_feats/spec
# datadir=/home/rostami/DS_10283_3336
# stage=3

# # . ./parse_options.sh 

# mkdir -p data/PA_train data/PA_dev data/PA_eval data/LA_train data/LA_dev data/LA_eval 

# if [ $stage -le 0 ]; then
#    echo "Stage 0: prepare dataset."
#    for access_type in PA LA;do
#        protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.train.trn.txt
#        awk '{print $2" "$4}' $protofile >data/${access_type}_train/utt2systemID 
#        awk '{print $2" "$1}' $protofile >data/${access_type}_train/utt2spk 
#        feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_train/utt2spk >data/${access_type}_train/spk2utt 
#        awk -v dir="${datadir}/${access_type}" -v type="${access_type}_train" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_train/wav.scp 
       
#        for dataset in dev eval;do
#            protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.${dataset}.trl.txt
#            awk '{print $2" "$4}' $protofile >data/${access_type}_${dataset}/utt2systemID 
#            awk '{print $2" "$1}' $protofile >data/${access_type}_${dataset}/utt2spk 
#            feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_${dataset}/utt2spk >data/${access_type}_${dataset}/spk2utt 
#            awk -v dir="${datadir}/${access_type}" -v type="${access_type}_${dataset}" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_${dataset}/wav.scp 
#        done
#     done
#     echo "dataset finished"
#    #   
# fi

# if [ $stage -le 1 ]; then
#    echo "Stage 1: extract Spec feats."
#    mkdir -p data/spec 
#    for name in PA_train PA_dev PA_eval LA_train LA_dev LA_eval; do
#        [ -d data/spec/${name} ]   cp -r data/${name} data/spec/${name} 
#        feats_extraction/make_spectrogram.sh --feats-config conf/feats/spec.conf --nj 80 --cmd "$train_cmd" \
#              data/spec/${name} exp/make_spec $specdir 
#    done
#    echo "Spec feats done"
#    #  
# fi

# if [ $stage -le 2 ]; then
#    echo "Stage 2: extract CQT feats."
#    python3 -W ignore feats_extraction/compute_CQT.py --out_dir /home/rostami/cqt --access_type PA --param_json_path conf/feats/cqt_48bpo_fmin15.json --num_workers 60 
#    # python3 feats_extraction/compute_CQT.py --out_dir data/cqt --access_type LA --param_json_path conf/feats/cqt_48bpo_fmin15.json --num_workers 60 
#    python3 -W ignore feats_extraction/GenLPCQTFeats_kaldi.py --access_type PA --work_dir /home/rostami/cqt 
#    # python3 feats_extraction/GenLPCQTFeats_kaldi.py --access_type LA --work_dir data/cqt 

#    # # uncomment below for removing numpy data to save space.
#    # find /home/rostami/cqt/ -maxdepth 2 -type f -name "*.npy" -delete
#    #  
# fi

# if [ $stage -le 3 ]; then
#    echo "Stage 3: truncate features and generate labels."
#    for name in PA_dev PA_train; do
#        for feat_type in cqt; do
#            echo "Processing $name $feat_type"
#            python3 feats_extraction/feat_slicing.py --in-scp /home/rostami/${feat_type}/${name}/feats.scp --out-scp /home/rostami/${feat_type}/${name}/feats_slicing.scp --out-ark /home/rostami/${feat_type}/${name}/feats_slicing.ark 
#             # echo "python3 feats_extraction/convertID2index.py --scp-file data/${feat_type}/${name}/feats_slicing.scp --sysID-file data/${feat_type}/${name}/utt2systemID --out-file data/${feat_type}/${name}/utt2index --access-type ${name:0:2}"
#          #   python3 feats_extraction/convertID2index.py --scp-file /home/rostami/${feat_type}/${name}/feats_slicing.scp --sysID-file /home/rostami/${feat_type}/${name}/utt2systemID --out-file /home/rostami/${feat_type}/${name}/utt2index --access-type ${name:0:2} 
#        done
#    done
# #     
# fi