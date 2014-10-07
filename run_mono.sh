#!/bin/sh

if [[ $# -lt 10 || $# -gt 12 ]]; then
  echo "`basename $0` remake outputDir trainFile lang dim numIters numThreads alpha neg lrOpt [minCount] [trainCount]" # [srcMonoFile tgtMonoFile monoSize anneal monoThread]
  echo "neg=0: use hierarchical softmax"
  echo "trainCount: number of tokens we will train (occurrences of words (>=minCount) + number of sentences)"
  exit
fi

remake=$1
outputDir=$2
trainFile=$3
lang=$4
dim=$5
numIter=$6
numThreads=$7
alpha=$8
neg=$9
lrOpt=${10}

minCountStr=""
if [ $# -ge 11 ]; then
  minCountStr="-min-count ${11}"
fi
trainCountStr=""
if [ $# -ge 12 ]; then
  trainCountStr="-src-train-count ${12}"
fi

monoStr=""
otherOpts=""
#if [ $# -eq 13 ]; then # mono
#  monoStr="-src-train-mono ${9} -tgt-train-mono ${10} -mono-size ${11} -anneal ${12} -mono-thread ${13}"
#  monoLambda=1
#  thresholdPerThread=-1
#  otherOpts="-monoLambda $monoLambda -threshold-per-thread $thresholdPerThread"
#fi

if [ $neg -gt 0 ]; then
  negStr="-negative $neg -hs 0"
else
  negStr="-negative 0 -hs 1"
fi
echo "negStr=$negStr"

echo "# monoStr=$monoStr"
echo "# minCountStr=$minCountStr"
echo "# trainCountStr=$trainCountStr"

if [ $remake -eq 1 ]
then
  make clean
  make
fi

VERBOSE=1
function execute_check {
  file=$1 
  cmd=$2
  
  if [[ -f $file || -d $file ]]; then
    echo ""
    echo "! File/directory $file exists. Skip."
  else
    echo ""
    if [ $VERBOSE -eq 1 ]; then
      echo "# Executing: $cmd"
    fi
    
    eval $cmd
  fi
}

# check outDir exists
echo "# outputDir=$outputDir"
execute_check $outputDir "mkdir -p $outputDir"

execute_check "" "cd ~/bivec"
execute_check "" "time ~/bivec/bivec -src-train $trainFile -src-lang $lang -output $outputDir/model -cbow 0 -size $dim -window 5 $negStr -sample 1e-5 -threads $numThreads -binary 0 -num-iters $numIter -eval 1 -alpha $alpha -lr-opt $lrOpt $minCountStr $trainCountStr $monoStr $otherOpts"
