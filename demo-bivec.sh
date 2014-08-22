make -f makefile.debug clean
make -f makefile.debug
if [ ! -d "output" ]; then
  mkdir output
fi

command="./bivec.debug -src-train data/data.10k.de -src-lang de -tgt-train data/data.10k.en -tgt-lang en -align data/data.10k.align -output vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-2 -threads 1 -binary 0 -eval 1 -num-iters 3"
echo "time $command"
time $command

