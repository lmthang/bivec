#!/bin/bash

if [[ $# -lt 8 || $# -gt 14 ]]; then
  echo "`basename $0` remake outputDir trainPrefix dim alignOpt numIters numThreads neg [isCbow alpha sample tgt_sample bi_weight otherOpts]" 
  echo "neg=0: use hierarchical softmax"
  exit
fi

srcLang="de"
tgtLang="en"

remake=$1
outputDir=$2
trainPrefix=$3
dim=$4
alignOpt=$5
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
src_sample="1e-5"
if [ $# -ge 11 ]; then
  src_sample=${11}
fi
tgt_sample="1e-5"
if [ $# -ge 12 ]; then
  tgt_sample=${12}
fi
bi_weight="1"
if [ $# -ge 13 ]; then
  bi_weight=${13}
fi
otherOpts=""
if [ $# -ge 14 ]; then
  otherOpts=${14}
fi


sampleStr="-sample $src_sample -tgt-sample $tgt_sample"
monoStr=""
otherOptStr="-bi-weight $bi_weight $otherOpts"

if [ $neg -gt 0 ]; then
  negStr="-negative $neg -hs 0"
else
  negStr="-negative 0 -hs 1"
fi
echo "negStr=$negStr"

name=`basename $trainPrefix`
echo "# isCbow=$isCbow"
echo "# alphaStr=$alphaStr"
echo "# sampleStr=$sampleStr"
echo "# monoStr=$monoStr"
echo "# otherOptStr=$otherOptStr"
echo "# name=$name"

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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $SCRIPT_DIR"
execute_check "" "cd $SCRIPT_DIR"
args="-src-train $trainPrefix.de -tgt-train $trainPrefix.en -src-lang $srcLang -tgt-lang $tgtLang -output $outputDir/out -cbow $isCbow -size $dim -window 5 $negStr -threads $numThreads -binary 0 -iter $numIter -eval 1 $alphaStr $sampleStr $monoStr $otherOptStr"

if [ $remake -eq 1 ]
then
  make clean
  make
fi

if [ $alignOpt -ge 1 ]; then
  execute_check "" "time $SCRIPT_DIR/bivec -align $trainPrefix.$srcLang-$tgtLang -align-opt $alignOpt $args"
else
  execute_check "" "time $SCRIPT_DIR/bivec $args"
fi
