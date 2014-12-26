make -f makefile clean
make -f makefile
if [ ! -d "output" ]; then
  mkdir output
fi

command="./text2vec -src-train data/data.10k.de -src-lang de -tgt-train data/data.10k.en -tgt-lang en -align data/data.10k.align -output vectors.bin -cbow 1 -size 200 -window 5 -negative 15 -hs 0 -sample 1e-2 -tgt-sample 1e-3 -threads 4 -binary 0 -eval 1 -iter 3"
echo "time $command"
time $command
