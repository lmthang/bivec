datapath=../../data

prefix=$1
#echo "en2de prepare data all sizes"
#echo [Preparing test set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/test/de --idf $datapath/idfs/idf.de --word-embeddings $prefix.de-en.de  --vector-file $prefix.doc.test.de-en.de

#echo [Preparing train set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --rnnlm --text-dir $datapath/rcv-from-binod/train/EN1000 --idf $datapath/idfs/idf.en --word-embeddings $prefix.de-en.en  --vector-file $prefix.doc.train.de-EN1000.en

