#!/bin/bash

# copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
feats_config=conf/spectrogram.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_spectrogram.sh [options] <data-dir> <log-dir> <path-to-spectrogramdir>";
   echo "options: "
   echo "  --spectrogram-config <config-file>                      # config passed to compute-spectrogram-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
spectrogramdir=$3


# make $spectrogramdir an absolute pathname.
spectrogramdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $spectrogramdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $spectrogramdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $feats_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_spectrogram.sh: no such file $f"
    exit 1;
  fi
done
# utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
fi

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=""
  for ((n=1; n<=nj; n++)); do
    split_segments="$split_segments $logdir/segments.$n"
  done
 
  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_spectrogram_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    compute-spectrogram-feats $vtln_opts --verbose=2 --config=$feats_config ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$spectrogramdir/raw_spectrogram_$name.JOB.ark,$spectrogramdir/raw_spectrogram_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for ((n=1; n<=nj; n++)); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;


  # add ,p to the input rspecifier so that we can just skip over
  # utterances that have bad wave data.

  echo "cmd: "$cmd
  echo "logdir: "$logdir
  $cmd JOB=1:$nj $logdir/make_spectrogram_${name}.JOB.log \
    compute-spectrogram-feats  $vtln_opts --verbose=2 --config=$feats_config \
     scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
      copy-feats --compress=$compress ark:- \
      ark,scp:$spectrogramdir/raw_spectrogram_$name.JOB.ark,$spectrogramdir/raw_spectrogram_$name.JOB.scp \
      || exit 1;
fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing spectrogram features for $name:"
  tail $logdir/make_spectrogram_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $spectrogramdir/raw_spectrogram_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating spectrogram features for $name"
