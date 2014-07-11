datapath=../../data

prefix=$1
echo "de2en prepare data train-valid"
#echo [Preparing test set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/test/DE1000_train_valid --idf $datapath/idfs/idf.de --word-embeddings $prefix.de-en.de  --vector-file $prefix.doc.test.DE1000_train_valid.de

#echo [Preparing train set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/train/DE1000_train_valid --idf $datapath/idfs/idf.de --word-embeddings $prefix.de-en.de  --vector-file $prefix.doc.train.DE1000_train_valid.de
