make clean
make
if [ ! -d "output" ]; then
  mkdir output
fi

args="-src-train data/data.10k.en -src-lang en -output output/vectors -cbow 1 -size 200 -window 5 -negative 5 -hs 0 -sample 1e-3 -threads 1 -binary 0 -eval 1-iter 3"
echo "time ./bivec $args"
time ./bivec $args 
