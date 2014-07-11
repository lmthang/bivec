datapath=../../data

prefix=$1
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.DE1000_train_valid.de  --model-name $prefix.classifiers.avperc.DE1000_train_valid.de   --epoch-num 10

java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.DE1000_train_valid.de  --model-name $prefix.classifiers.avperc.DE1000_train_valid.de


