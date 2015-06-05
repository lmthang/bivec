This code is based on Mikolov's word2vec, version r42 https://code.google.com/p/word2vec/source/detail?r=42.
It has all the functionalities of word2vec with the following added features:
  (a) Train bilingual embeddings as described in the paper "Bilingual Word Representations with Monolingual Quality in Mind".
  (b) When training bilingual embeddings for English and German, it automatically produces the cross-lingual document classification results.
  (c) For monolingual embeddings, the code outputs word similarity results for English, German and word analogy results for English.
  (d) Save output vectors besides input vectors.
  (e) Automatically save vocab file and load vocab (if there's one exists).
  (f) The code has been extensively refactored to make it easier to understand and more comments have been added.

If you use this software, please cite this paper:
@inproceedings{Luong-etal:naacl15:bivec,
        Address = {Denver, United States}
        Author = {Luong, Minh-Thang  and  Pham, Hieu and Manning, Christopher D.},
        Booktitle = {NAACL Workshop on Vector Space Modeling for NLP},
        Title = {Bilingual Word Representations with Monolingual Quality in Mind},
        Year = {2015}}

Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
  with many contributions from Hieu Pham <hyhieu@stanford.edu>

Files & Directories:
(a) demo-bi-*: test various bilingual models.
(b) demo-mono-*: test monolingual models.
(c) wordsim / analogy: code to evaluate trained embeddings on the word similarity and analogy tasks.
(e) run_mono.sh: train mono models.
(f) run_bi.sh: train bilingual embedding models (Note: in this script, we hard-coded the source language to be "de" and the source language to be "en".)
(g) cldc/: cross-lingual document classification (CLDC) task. 
  To be able to obtain the CLDC results during training of the bilingual embeddings, you need the following:
  (i) put under cldc/, the following two directories: src/ for the perceptron code and data/ for the task. These two directories can be obtained from the authors of this paper "Inducing crosslingual distributed rep- resentations of words".
  (ii) go into cldc/, and run ant

Notes:
If you don't have Matlab, modify demo-*.sh to set -eval 0 (instead of -eval 1).

Sample commands:
* Bi model: run_bi.sh remake outputDir trainPrefix dim alignOpt numIters numThreads neg [isCbow alpha sample tgt_sample bi_weight otherOpts]
./run_bi.sh 1 outputDir data/data.10k 50 1 5 4 10
* Mono model: run_mono.sh remake outputDir trainFile lang dim numIters numThreads neg [otherOpts]
./run_mono.sh 1 outputDir data/data.10k.en en 50 5 2 5


