datapath=../../data

prefix=$1
#echo "Training on EN100"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN100.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

#echo "Training on EN200"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN200.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

#echo "Training on EN500"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN500.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

#echo "Training on EN1000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN1000.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

#echo "Training on EN5000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN5000.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 

#echo "Training on EN10000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $prefix.doc.train.de-EN10000.en  --model-name $prefix.classifiers.avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $prefix.doc.test.de-en.de  --model-name $prefix.classifiers.avperc.de-en.de 



