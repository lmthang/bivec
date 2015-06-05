make -f makefile clean
make -f makefile
if [ ! -d "output" ]; then
  mkdir output
fi

command="./bivec -src-train data/data.10k.de -src-lang de -tgt-train data/data.10k.en -tgt-lang en -align data/data.10k.align -output output/vectors -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -tgt-sample 1e-3 -threads 1 -binary 0 -eval 1 -iter 3 -bi-weight 1.0 -align-opt 1"
echo "time $command"
time $command

