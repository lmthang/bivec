#!/bin/sh

if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "`basename $0` remake outputDir trainPrefix dim useAlign numIters numThreads neg [isCbow alpha]" # [srcMonoFile tgtMonoFile monoSize anneal monoThread]
  echo "neg=0: use hierarchical softmax"
  exit
fi

remake=$1
outputDir=$2
trainPrefix=$3
dim=$4
useAlign=$5
numIter=${6}
numThreads=${7}
neg=$8
isCbow=0
if [ $# -ge 9 ]; then
  isCbow=${9}
fi
alphaStr=""
if [ $# -ge 10 ]; then
  alphaStr="-alpha ${10}"
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

name=`basename $trainPrefix`
echo "# isCbow=$isCbow"
echo "# alphaStr=$alphaStr"
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

execute_check "" "cd ~/text2vec"

if [ $useAlign -eq 1 ]
then
  execute_check "" "time ~/text2vec/text2vec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -align $trainPrefix.de-en -src-lang de -tgt-lang en -output $outputDir/out -cbow $isCbow -size $dim -window 5 $negStr -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -eval 1 $alphaStr $monoStr $otherOpts"
else
  execute_check "" "time ~/text2vec/text2vec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -src-lang de -tgt-lang en -output $outputDir/out -cbow $isCbow -size $dim -window 5 $negStr -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -eval 1 $alphaStr $monoStr $otherOpts"
fi
