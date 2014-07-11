#!/bin/sh

cd ~/bi_word2vec/

if [ $# -ne 8 ]
then
  echo "`basename $0` remake trainSize dim biTrain useAlign logEvery numIters numThreads"
  exit
fi

remake=$1

trainSize=$2
dim=$3
biTrain=$4
useAlign=$5
logEvery=$6

numIter=$7
numThreads=$8

outputDir=$trainSize.$dim.$biTrain.$useAlign

if [ $remake -eq 1 ]
then
  make clean
  make
fi

if [ ! -d "output" ]
then
  mkdir output
fi

if [ ! -d "output/$outputDir" ]
then
  mkdir output/$outputDir
fi

if [ $useAlign -eq 1 ]
then
  echo "time ./word2vec -src-train ../data/hieu/data.$trainSize.de -tgt-train ../data/hieu/data.$trainSize.en -align ../data/hieu/data.$trainSize.align -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery"
  
  time ./word2vec -src-train ../data/hieu/data.$trainSize.de -tgt-train ../data/hieu/data.$trainSize.en -align ../data/hieu/data.$trainSize.align -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery
else
  echo "time ./word2vec -src-train ../data/hieu/data.$trainSize.de -tgt-train ../data/hieu/data.$trainSize.en -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery"
  
  time ./word2vec -src-train ../data/hieu/data.$trainSize.de -tgt-train ../data/hieu/data.$trainSize.en -src-lang de -tgt-lang en -output output/$outputDir/out.$trainSize.$dim.$biTrain -cbow 0 -size $dim -window 5 -negative 5 -hs 0 -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -biTrain $biTrain -logEvery $logEvery
fi
