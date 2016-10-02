#!/bin/sh

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "`basename $0` remake outputDir trainFile lang dim numIters numThreads neg [otherOpts]"
  echo "remake=1: to re-make the code again"
  echo "neg=0: use hierarchical softmax"
  exit
fi

remake=$1
outputDir=$2
trainFile=$3
lang=$4
dim=$5
numIter=$6
numThreads=$7
neg=$8
otherOpts=""
if [ $# -ge 9 ]; then
  otherOpts=${9}
fi

if [ $neg -gt 0 ]; then
  negStr="-negative $neg -hs 0"
else
  negStr="-negative 0 -hs 1"
fi
echo "negStr=$negStr"

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

if [ $remake -eq 1 ]
then
  make clean
  make
fi

execute_check "" "time $SCRIPT_DIR/bivec -src-train $trainFile -src-lang $lang -output $outputDir/model -cbow 0 -size $dim -window 5 $negStr -sample 1e-5 -threads $numThreads -binary 0 -iter $numIter -eval 1 $otherOpts"
