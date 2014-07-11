datapath=../../data

prefix=$1
#echo "Training on DE1000 $prefix"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.en-DE1000.de  --model-name $prefix.classifiers.avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.en-de.en  --model-name $prefix.classifiers.avperc.en-de.en 

