#!/bin/sh

if [[ $# -ne 9 && $# -ne 14 ]]; then
  echo "`basename $0` remake outPrefix trainPrefix dim biTrain useAlign logEvery numIters numThreads [srcMonoFile tgtMonoFile monoSize anneal monoThread]"
  exit
fi

remake=$1
outputDir=$2
trainPrefix=$3
dim=$4
biTrain=$5
useAlign=$6
logEvery=$7
numIter=${8}
numThreads=${9}

monoStr=""
if [ $# -eq 14 ]; then # mono
  monoStr="-src-train-mono ${10} -tgt-train-mono ${11} -mono-size ${12} -anneal ${13} -mono-thread ${14}"
fi

name=`basename $trainPrefix`
echo "# monoStr=$monoStr"
echo "# name=$name"

#srcMonoPartial=$7
#tgtMonoPartial=${8}
#outputDir=$name.$dim.$biTrain.$useAlign.mono$monoSize

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
echo "# outputDir=$outputDir"
execute_check $outputDir "mkdir -p $outputDir"

if [ $useAlign -eq 1 ]
then
  execute_check "" "time ./word2vec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -align $trainPrefix.de-en -src-lang de -tgt-lang en -output $outputDir/out -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery $monoStr $otherOpts"
else
  execute_check "" "time ./word2vec -src-train $trainPrefix.de -tgt-train $trainPrefix.en -src-lang de -tgt-lang en -output $outputDir/out -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery $monoStr $otherOpts"
fi
#-src-mono-partial $srcMonoPartial -tgt-mono-partial $tgtMonoPartial
