make clean
make
if [ ! -d "output" ]; then
  mkdir output
fi

DATA=data/data.10k.en #"/Users/lmthang/RA/sentiment/aclImdb/preprocessed/train.text.tok" #
args="-src-train $DATA -src-lang en -output output/vectors -cbow 1 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 1 -binary 1 -eval 0"
echo "time ./bivec $args"
time ./bivec $args 

echo "./distance output/vectors.en"
./distance output/vectors.en
