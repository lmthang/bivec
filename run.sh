#!/bin/sh

if [[ $# -ne 8 && $# -ne 13 ]]; then
  echo "`basename $0` remake trainSize dim biTrain useAlign logEvery numIters numThreads [srcMonoFile tgtMonoFile monoSize anneal monoThread]"
  exit
fi

remake=$1

trainSize=$2
dim=$3
biTrain=$4
useAlign=$5
logEvery=$6
numIter=${7}
numThreads=${8}

monoStr=""
if [ $# -eq 13 ]; then # mono
  monoStr="-src-train-mono $9 -tgt-train-mono ${10} -mono-size ${11} -anneal ${12} -mono-thread ${13}"
fi
echo "# monoStr=$monoStr"

#srcMonoPartial=$7
#tgtMonoPartial=${8}

outputDir=$trainSize.$dim.$biTrain.$useAlign.mono$monoSize

monoLambda=1
thresholdPerThread=-1
otherOpts="-monoLambda $monoLambda -threshold-per-thread $thresholdPerThread"
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
execute_check output/$outDir "mkdir -p output/$outDir"

if [ $useAlign -eq 1 ]
then
  execute_check "" "time ./word2vec -src-train data/data.$trainSize.de -tgt-train data/data.$trainSize.en -align data/data.$trainSize.align -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery $monoStr $otherOpts"
else
  execute_check "" "time ./word2vec -src-train data/data.$trainSize.de -tgt-train data/data.$trainSize.en -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery $monoStr $otherOpts"
fi
#-src-mono-partial $srcMonoPartial -tgt-mono-partial $tgtMonoPartial
