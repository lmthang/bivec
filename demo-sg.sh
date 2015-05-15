make clean
make
if [ ! -d "output" ]; then
  mkdir output
fi

command="./text2vec -src-train data/data.10k.en -src-lang en -output vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 1 -binary 0 -eval 1"
echo "time $command"
time $command

