
mkdir -p logs || exit 1

randomseed=1010 # 0, 1, 2, ...
config=conf/training_mdl/abn_pa.json # configuration files in conf/training_mdl
feats=pa_cqt  # `pa_spec`, `pa_cqt`, `pa_lfcc`, `la_spec`, `la_cqt` or `la_lfcc`
runid=abn_pa_cqt_0.005_specaug_fixlbl_mean_seres_weighted
echo "Start training."
python3 train.py --run-id $runid --random-seed $randomseed --data-feats $feats --configfile $config > "logs/${runid}.txt"

echo "Start evaluation on all checkpoints."
# for model in `ls model_snapshots/$runid/ | sort -V -r`; do
#     echo $model
#     python3 eval.py --run-id $runid --random-seed $randomseed --data-feats $feats --configfile $config --pretrained "model_snapshots/${runid}/${model}"
# done

