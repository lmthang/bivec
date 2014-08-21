#!/bin/sh

if [[ $# -ne 7 && $# -ne 12 ]]; then
  echo "`basename $0` remake outPrefix trainPrefix dim useAlign numIters numThreads [srcMonoFile tgtMonoFile monoSize anneal monoThread]"
  exit
fi

remake=$1
outputDir=$2
trainPrefix=$3
dim=$4
useAlign=$5
numIter=${6}
numThreads=${7}

monoStr=""
otherOpts=""
if [ $# -eq 11 ]; then # mono
  monoStr="-src-train-mono ${8} -tgt-train-mono ${9} -mono-size ${10} -anneal ${11} -mono-thread ${12}"
  monoLambda=1
  thresholdPerThread=-1
  otherOpts="-monoLambda $monoLambda -threshold-per-thread $thresholdPerThread"
fi

name=`basename $trainPrefix`
echo "# monoStr=$monoStr"
echo "# name=$name"

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

if [ $useAlign -eq 1 ]
then
  execute_check "" "time ~/bivec/bivec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -align $trainPrefix.de-en -src-lang de -tgt-lang en -output $outputDir/out -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -num-iters $numIter $monoStr $otherOpts"
else
  execute_check "" "time ~/bivec/bivec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -src-lang de -tgt-lang en -output $outputDir/out -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -num-iters $numIter $monoStr $otherOpts"
fi
