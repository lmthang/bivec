datapath=../../data

prefix=$1
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.EN1000_train_valid.en  --model-name $prefix.classifiers.avperc.EN1000_train_valid.en   --epoch-num 10

java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.EN1000_train_valid.en  --model-name $prefix.classifiers.avperc.EN1000_train_valid.en


