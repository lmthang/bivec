datapath=../../data

prefix=$1
#echo "Training on EN1000 $prefix"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN1000.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

