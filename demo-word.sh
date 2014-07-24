make clean
make
#if [ ! -e text8 ]; then
#  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
#  gzip -d text8.gz -f
#fi
#time ./word2vec -train text8 -output vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1
if [ ! -d "output" ]; then
  mkdir output
fi

time ./word2vec -src-train data/data.10k.de -tgt-train data/data.10k.en -align data/data.10k.align -src-lang de -tgt-lang en -output output/out -cbow 0 -size 40 -window 5 -negative 5 -hs 0 -sample 1e-5 -threads 10 -binary 0 -iter 10 -biTrain 0
#time ./word2vec -src-train data/data.de-en.de -tgt-train data/data.de-en.en -align data/data.de-en.align -src-lang de -tgt-lang en -output output/out -cbow 0 -size 200 -window 5 -negative 5 -hs 0 -sample 1e-5 -threads 4 -binary 0 -iter 10 
#time ./word2vec -src-train data/data.zh-en.zh -tgt-train data/data.zh-en.en -align data/data.zh-en.align -src-lang zh -tgt-lang en -output output/out -cbow 0 -size 200 -window 5 -negative 5 -hs 0 -sample 1e-5 -threads 1 -binary 0 -iter 1 

