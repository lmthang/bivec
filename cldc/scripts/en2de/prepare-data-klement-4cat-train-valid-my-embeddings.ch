datapath=../../data

prefix=$1
echo "en2de prepare data train-valid"
#echo [Preparing test set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/test/EN1000_train_valid --idf $datapath/idfs/idf.en --word-embeddings $prefix.de-en.en  --vector-file $prefix.doc.test.EN1000_train_valid.en

#echo [Preparing train set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/train/EN1000_train_valid --idf $datapath/idfs/idf.en --word-embeddings $prefix.de-en.en  --vector-file $prefix.doc.train.EN1000_train_valid.en
