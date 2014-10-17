text2vec
Thang Luong @ 2014, <lmthang@stanford.edu>. With contributions from Hieu Pham <hyhieu@stanford.edu>.

This codebase is based on the word2vec code by Tomas Mikolov
https://code.google.com/p/word2vec/ with added functionalities:
 (a) Train multiple iterations
 (b) Save in/out vectors
 (c) wordsim/analogy evaluation
 (d) Automatically save vocab file and load vocab if there's one exists.
 (e) More comments

Files & Directories:
(a) demo-skip.sh: test skip-gram model 
(b) demo-cbow.sh: test cbow model
(c) demo/: contains expected outputs of the above scripts
(d) wordsim / analogy: to evaluate trained embeddings on the word similarity
and analogy tasks.
(e) run_mono.sh: train mono models.
(f) cldc/, run_bi.sh, demo-bi.sh: related to bilingual embedding models (on-going work). To be able to run these, you need to do:
  (i) go into cldc/, and run ant
  (ii) and copy the CLDC's data into cldc/data (ask Thang/Hieu for such data).

Notes:
If you don't have Matlab, modify demo-* to set -eval 0 (instead of -eval 1).

-------- Mikolov's README -------

Tools for computing distributed representtion of words
------------------------------------------------------

We provide an implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous
Bag-of-Words or the Skip-Gram neural network architectures. The user should to specify the following:
 - desired vector dimensionality
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model
 - training algorithm: hierarchical softmax and / or negative sampling
 - threshold for downsampling the frequent words 
 - number of threads to use
 - the format of the output word vector file (text or binary)

Usually, the other hyper-parameters such as the learning rate do not need to be tuned for different training sets. 

The script demo-word.sh downloads a small (100MB) text corpus from the web, and trains a small word vector model. After the training
is finished, the user can interactively explore the similarity of the words.

More information about the scripts is provided at https://code.google.com/p/word2vec/

