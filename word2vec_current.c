//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// Thang May14
#include <unistd.h> // for chdir
#include <assert.h>
#include <ctype.h>
//#include <limits.h>

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_LINE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_SENT_LENGTH 100 // Thang May14

typedef float real;                    // Precision of float numbers

const int vocab_hash_size = 50000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

const real beta = 1.1;

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

real src_mono_partial = 1.0, tgt_mono_partial = 1.0;
long long *src_mono_line_blocks, *tgt_mono_line_blocks;
long long *src_mono_line_blocks_pos, *tgt_mono_line_blocks_pos;
long long src_mono_line_count=0, tgt_mono_line_count=0, parallel_line_count=0, threshold_per_thread = -1;
real left_threshold, right_threshold;
char src_train_mono[MAX_STRING], tgt_train_mono[MAX_STRING];
int biTrain, useAlign, logEvery;
real monoLambda = 1.0;
int hs = 1, negative = 0;
const int table_size = 1e8;
clock_t start;
real alpha = 0.025, starting_alpha, sample = 0;

// Thang: number of training iterations
int num_train_iters = 1;
int start_iter = 0;
char align_file[MAX_STRING];
long long *align_line_blocks;
real align_sample = 0;
char output_prefix[MAX_STRING];
long long mono_size=-1; // number of sentences used for each mono corpus, override src/tgt_mono_partial
long long parallel_sent_count_actual=0, src_sent_count_actual=0, tgt_sent_count_actual=0;
real anneal = 1;
int cur_iter = 0;
int mono_thread = 0;
int src_num_threads, tgt_num_threads, parallel_num_threads;

/*******************************
 * src parametrs and functions *
 *******************************/
char src_train_file[MAX_STRING], src_output_file[MAX_STRING], src_lang[MAX_STRING];
char src_save_vocab_file[MAX_STRING], src_read_vocab_file[MAX_STRING];
struct vocab_word *src_vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *src_vocab_hash;
long long src_vocab_max_size = 1000, src_vocab_size = 0, layer1_size = 100;
long long src_train_words = 0, src_file_size = 0, classes = 0;
real *src_syn0, *src_syn1, *src_syn1neg, *expTable;
long long *src_line_blocks;
int *src_table;
// Thang: to load previously trained word vectors
char src_word_vector_file[MAX_STRING];
int is_src_word_vector_file = 0;
int src_is_train = 1; // if src_word_vector_file is specified, src_is_train = 0
long long src_parallel_train_words = 0;
long long src_mono_train_words = 0;
//long long src_parallel_word_count_actual = 0;

/******************
 * tgt parameters *
 ******************/
char tgt_train_file[MAX_STRING], tgt_output_file[MAX_STRING], tgt_lang[MAX_STRING];
char tgt_save_vocab_file[MAX_STRING], tgt_read_vocab_file[MAX_STRING];
struct vocab_word *tgt_vocab;
int *tgt_vocab_hash;
long long tgt_vocab_max_size = 1000, tgt_vocab_size = 0;
long long tgt_train_words = 0, tgt_file_size = 0;
real *tgt_syn0, *tgt_syn1, *tgt_syn1neg, *expTable;
long long *tgt_line_blocks;
int *tgt_table;
// Thang: to load previously trained word vectors
char tgt_word_vector_file[MAX_STRING];
int is_tgt_word_vector_file = 0;
int tgt_is_train = 1; // if tgt_word_vector_file is specified, tgt_is_train = 0
long long tgt_parallel_train_words = 0;
long long tgt_mono_train_words = 0;
//long long tgt_parallel_word_count_actual = 0;
/*****************
 * src functions *
 *****************/
void src_InitUnigramTable() {
  int a, i;
  long long src_train_words_pow = 0;
  real d1, power = 0.75;
  src_table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < src_vocab_size; a++) src_train_words_pow += pow(src_vocab[a].cn, power);
  i = 0;
  d1 = pow(src_vocab[i].cn, power) / (real)src_train_words_pow;
  for (a = 0; a < table_size; a++) {
    src_table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(src_vocab[i].cn, power) / (real)src_train_words_pow;
    }
    if (i >= src_vocab_size) i = src_vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void src_ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int src_GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int src_SearchVocab(char *word) {
  unsigned int hash = src_GetWordHash(word);
  while (1) {
    if (src_vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, src_vocab[src_vocab_hash[hash]].word)) return src_vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int src_ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  src_ReadWord(word, fin);
  if (feof(fin)) return -1;
  return src_SearchVocab(word);
}

// Adds a word to the vocabulary
int src_AddWordToVocab(char *word) {
  //  if (strcmp(word, "umfassen") == 0) fprintf(stderr, ">>>>>>> HERE\n");
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  //  if (strcmp(word, "umfassen") == 0) fprintf(stderr, ">>>>>>> HERE next %d %d\n", src_vocab_size, src_vocab_max_size);
  src_vocab[src_vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(src_vocab[src_vocab_size].word, word);
  src_vocab[src_vocab_size].cn = 0;
  src_vocab_size++;
  // Reallocate memory if needed
  //  if (strcmp(word, "umfassen") == 0) fprintf(stderr, ">>>>>>> still ok\n");
  if (src_vocab_size + 2 >= src_vocab_max_size) {
    src_vocab_max_size += 1000;
    src_vocab = (struct vocab_word *)realloc(src_vocab, src_vocab_max_size * sizeof(struct vocab_word));
  }
  hash = src_GetWordHash(word);
  while (src_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  src_vocab_hash[hash] = src_vocab_size - 1;
  return src_vocab_size - 1;
}

// Used later for sorting by word counts
int src_VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void src_SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&src_vocab[1], src_vocab_size - 1, sizeof(struct vocab_word), src_VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) src_vocab_hash[a] = -1;
  size = src_vocab_size;
  src_train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (src_vocab[a].cn < min_count) {
      src_vocab_size--;
      free(src_vocab[src_vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=src_GetWordHash(src_vocab[a].word);
      while (src_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      src_vocab_hash[hash] = a;
      src_train_words += src_vocab[a].cn;
    }
  }
  src_vocab = (struct vocab_word *)realloc(src_vocab, (src_vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < src_vocab_size; a++) {
    src_vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    src_vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void src_ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < src_vocab_size; a++) if (src_vocab[a].cn > min_reduce) {
    src_vocab[b].cn = src_vocab[a].cn;
    src_vocab[b].word = src_vocab[a].word;
    b++;
  } else free(src_vocab[a].word);
  src_vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) src_vocab_hash[a] = -1;
  for (a = 0; a < src_vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = src_GetWordHash(src_vocab[a].word);
    while (src_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    src_vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Thang Jun: add num_lines, only load vocab up to this amount of lines
void src_LearnVocabFromTrainFile(char *train_file, long long num_lines) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  if (strcmp(train_file, src_train_file) == 0) {
    for (a = 0; a < vocab_hash_size; a++) src_vocab_hash[a] = -1;
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: src training data file %s not found!\n", train_file);
    exit(1);
  }
  src_AddWordToVocab((char *)"</s>");

  // Thang
  long long line_count = 0, end_index;
  end_index = src_SearchVocab((char *)"</s>");
  fprintf(stderr, "# Load vocab from %s, only read num_lines=%lld, end_index=%lld\n", train_file, num_lines, end_index);
  while (1) {
    // Thang: check if we should stop after reading num_lines
    if(line_count == num_lines) { // Thang: we put at the top, to consider for the case when num_lines=0
      fprintf(stderr, "  stop after reading %lld lines\n", num_lines);
      break;
    }

    src_ReadWord(word, fin);
    if (feof(fin)) break;
    src_train_words++;
    if ((debug_mode > 1) && (src_train_words % 100000 == 0)) {
      printf("%lldK%c", src_train_words / 1000, 13);
      fflush(stdout);
    }
    i = src_SearchVocab(word);
    //    if (strcmp(train_file, src_train_mono) == 0) fprintf(stderr, "%s ---> %lld\n", word, i);
    if (i == -1) {
      a = src_AddWordToVocab(word);
      src_vocab[a].cn = 1;
    } else src_vocab[i].cn++;
    if (src_vocab_size > vocab_hash_size * 0.7) src_ReduceVocab();

    // Thang: check if we should stop after reading num_lines
    if(i==end_index) { // end of line
      ++line_count;
    }
  }
  if (debug_mode > 0) {
    printf("  Source learn vocab size: %lld\n", src_vocab_size);
    printf("  Words in source train file: %lld\n", src_train_words);
    fprintf(stderr, "  Source learn vocab size: %lld\n", src_vocab_size);
    fprintf(stderr, "  Words in source train file: %lld\n", src_train_words);
  }
  src_file_size = ftell(fin);
  fclose(fin);
}

void src_SaveVocab() {
  long long i;
  FILE *fo = fopen(src_save_vocab_file, "wb");
  for (i = 0; i < src_vocab_size; i++) fprintf(fo, "%s %lld\n", src_vocab[i].word, src_vocab[i].cn);
  fclose(fo);
}

void src_ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(src_read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) src_vocab_hash[a] = -1;
  src_vocab_size = 0;
  while (1) {
    src_ReadWord(word, fin);
    if (feof(fin)) break;
    a = src_AddWordToVocab(word);
    fscanf(fin, "%lld%c", &src_vocab[a].cn, &c);
    i++;
  }
  src_SortVocab();
  if (debug_mode > 0) {
    printf("Source read vocab size: %lld\n", src_vocab_size);
    printf("Words in source train file: %lld\n", src_train_words);
  }
  fin = fopen(src_train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  src_file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&src_syn0, 128, (long long)src_vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **)&tgt_syn0, 128, (long long)tgt_vocab_size * layer1_size * sizeof(real));
  if (src_syn0 == NULL || tgt_syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (negative>0) {
    a = posix_memalign((void **)&src_syn1neg, 128, (long long)src_vocab_size * layer1_size * sizeof(real));
    if (src_syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < src_vocab_size; a++)
      src_syn1neg[a * layer1_size + b] = 0;
    a = posix_memalign((void **)&tgt_syn1neg, 128, (long long)tgt_vocab_size * layer1_size * sizeof(real));
    if (tgt_syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < tgt_vocab_size; a++)
      tgt_syn1neg[a * layer1_size + b] = 0;
  }
  for (b = 0; b < layer1_size; b++) for (a = 0; a < src_vocab_size; a++)
    src_syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  for (b = 0; b < layer1_size; b++) for (a = 0; a < tgt_vocab_size; a++)
    tgt_syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  //  src_CreateBinaryTree();
}




/*******************************
 * tgt functions *
 *******************************/

void tgt_InitUnigramTable() {
  int a, i;
  long long tgt_train_words_pow = 0;
  real d1, power = 0.75;
  tgt_table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < tgt_vocab_size; a++) tgt_train_words_pow += pow(tgt_vocab[a].cn, power);
  i = 0;
  d1 = pow(tgt_vocab[i].cn, power) / (real)tgt_train_words_pow;
  for (a = 0; a < table_size; a++) {
    tgt_table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(tgt_vocab[i].cn, power) / (real)tgt_train_words_pow;
    }
    if (i >= tgt_vocab_size) i = tgt_vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void tgt_ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int tgt_GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int tgt_SearchVocab(char *word) {
  unsigned int hash = tgt_GetWordHash(word);
  while (1) {
    if (tgt_vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, tgt_vocab[tgt_vocab_hash[hash]].word)) return tgt_vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int tgt_ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  tgt_ReadWord(word, fin);
  if (feof(fin)) return -1;
  return tgt_SearchVocab(word);
}

// Adds a word to the vocabulary
int tgt_AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  tgt_vocab[tgt_vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(tgt_vocab[tgt_vocab_size].word, word);
  tgt_vocab[tgt_vocab_size].cn = 0;
  tgt_vocab_size++;
  // Reallocate memory if needed
  if (tgt_vocab_size + 2 >= tgt_vocab_max_size) {
    tgt_vocab_max_size += 1000;
    tgt_vocab = (struct vocab_word *)realloc(tgt_vocab, tgt_vocab_max_size * sizeof(struct vocab_word));
  }
  hash = tgt_GetWordHash(word);
  while (tgt_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  tgt_vocab_hash[hash] = tgt_vocab_size - 1;
  return tgt_vocab_size - 1;
}

// Used later for sorting by word counts
int tgt_VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void tgt_SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&tgt_vocab[1], tgt_vocab_size - 1, sizeof(struct vocab_word), tgt_VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) tgt_vocab_hash[a] = -1;
  size = tgt_vocab_size;
  tgt_train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (tgt_vocab[a].cn < min_count) {
      tgt_vocab_size--;
      free(tgt_vocab[tgt_vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=tgt_GetWordHash(tgt_vocab[a].word);
      while (tgt_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      tgt_vocab_hash[hash] = a;
      tgt_train_words += tgt_vocab[a].cn;
    }
  }
  tgt_vocab = (struct vocab_word *)realloc(tgt_vocab, (tgt_vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < tgt_vocab_size; a++) {
    tgt_vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    tgt_vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void tgt_ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < tgt_vocab_size; a++) if (tgt_vocab[a].cn > min_reduce) {
    tgt_vocab[b].cn = tgt_vocab[a].cn;
    tgt_vocab[b].word = tgt_vocab[a].word;
    b++;
  } else free(tgt_vocab[a].word);
  tgt_vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) tgt_vocab_hash[a] = -1;
  for (a = 0; a < tgt_vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = tgt_GetWordHash(tgt_vocab[a].word);
    while (tgt_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    tgt_vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void tgt_CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(tgt_vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(tgt_vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(tgt_vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < tgt_vocab_size; a++) count[a] = tgt_vocab[a].cn;
  for (a = tgt_vocab_size; a < tgt_vocab_size * 2; a++) count[a] = 1e15;
  pos1 = tgt_vocab_size - 1;
  pos2 = tgt_vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < tgt_vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[tgt_vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = tgt_vocab_size + a;
    parent_node[min2i] = tgt_vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < tgt_vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == tgt_vocab_size * 2 - 2) break;
    }
    tgt_vocab[a].codelen = i;
    tgt_vocab[a].point[0] = tgt_vocab_size - 2;
    for (b = 0; b < i; b++) {
      tgt_vocab[a].code[i - b - 1] = code[b];
      tgt_vocab[a].point[i - b] = point[b] - tgt_vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void tgt_LearnVocabFromTrainFile(char *train_file, long long num_lines) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) tgt_vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: tgt training data file not found!\n");
    exit(1);
  }
  //  tgt_vocab_size = 0;
  tgt_AddWordToVocab((char *)"</s>");

  // Thang
  long long line_count = 0, end_index;
  end_index = tgt_SearchVocab((char *)"</s>");
  fprintf(stderr, "# Load vocab from %s, only read num_lines=%lld, end_index=%lld\n", train_file, num_lines, end_index);
  while (1) {
    // Thang: check if we should stop after reading num_lines
    if(line_count == num_lines) { // Thang: we put at the top, to consider for the case when num_lines=0
      fprintf(stderr, "  stop after reading %lld lines\n", num_lines);
      break;
    }

    tgt_ReadWord(word, fin);
    if (feof(fin)) break;
    tgt_train_words++;
    if ((debug_mode > 1) && (tgt_train_words % 100000 == 0)) {
      printf("%lldK%c", tgt_train_words / 1000, 13);
      fflush(stdout);
    }
    i = tgt_SearchVocab(word);
    if (i == -1) {
      a = tgt_AddWordToVocab(word);
      tgt_vocab[a].cn = 1;
    } else tgt_vocab[i].cn++;
    if (tgt_vocab_size > vocab_hash_size * 0.7) tgt_ReduceVocab();

    if(i==end_index) { // end of line
      ++line_count;
    }
  }
  //  tgt_SortVocab();
  if (debug_mode > 0) {
    printf("  Target learn vocab size: %lld\n", tgt_vocab_size);
    printf("  Words in target train file: %lld\n", tgt_train_words);
    fprintf(stderr, "  Source learn vocab size: %lld\n", tgt_vocab_size);
    fprintf(stderr, "  Words in source train file: %lld\n", tgt_train_words);
  }
  tgt_file_size = ftell(fin);
  fclose(fin);
}

void tgt_SaveVocab() {
  long long i;
  FILE *fo = fopen(tgt_save_vocab_file, "wb");
  for (i = 0; i < tgt_vocab_size; i++) fprintf(fo, "%s %lld\n", tgt_vocab[i].word, tgt_vocab[i].cn);
  fclose(fo);
}

void tgt_ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(tgt_read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) tgt_vocab_hash[a] = -1;
  tgt_vocab_size = 0;
  while (1) {
    tgt_ReadWord(word, fin);
    if (feof(fin)) break;
    a = tgt_AddWordToVocab(word);
    fscanf(fin, "%lld%c", &tgt_vocab[a].cn, &c);
    i++;
  }
  tgt_SortVocab();
  if (debug_mode > 0) {
    printf("Target read vocab size: %lld\n", tgt_vocab_size);
    printf("Words in train file: %lld\n", tgt_train_words);
  }
  fin = fopen(tgt_train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  tgt_file_size = ftell(fin);
  fclose(fin);
}

// Hieu: read the file file_name, return the num_blocks starting points in blocks[]
void ComputeBlockStartPoints(char* file_name, int num_blocks, long long **blocks, long long *num_lines) {
  fprintf(stderr, "# ComputeBlockStartPoints %s ... ", file_name);
  long long curr_line = 0, block_size;
  int line_count = 0;
  int curr_block = 0;
  int i;
  char line[MAX_LINE_LENGTH];
  FILE *file;

  *num_lines = 0;
  file = fopen(file_name, "r");
  while (1) {
    fgets(line, MAX_LINE_LENGTH, file);
    if (feof(file)) {
      break;
    }
    ++(*num_lines);
  }

  if (strcmp(file_name, src_train_mono)==0) {
    if(mono_size>=0 && mono_size<(*num_lines)){ // use specific size
      *num_lines = mono_size;
    } else { // use proportion
      *num_lines = (*num_lines) * src_mono_partial;
    }
  }
  if (strcmp(file_name, tgt_train_mono)==0) {
    if(mono_size>=0 && mono_size<(*num_lines)){ // use specific size
      *num_lines = mono_size;
    } else { // use proportion
      *num_lines = (*num_lines) * tgt_mono_partial;
    }
  }

  fseek(file, 0, SEEK_SET);
  block_size = (*num_lines + num_blocks - 1) / num_blocks;
  *blocks = malloc((num_blocks+1) * sizeof(long long));
  (*blocks)[0] = 0;
  curr_block = 0;
  while (1) {
    fgets(line, MAX_LINE_LENGTH, file);
    if (feof(file)) {
      break;
    }

    if (++curr_line == block_size || ++line_count==(*num_lines)) { // Thang: add check line_count
      curr_line = 0;
      if (++curr_block < num_blocks) (*blocks)[curr_block] = (long long)ftell(file);
      else break;
    }

    // Thang
    if (curr_block==num_blocks) break;
  }
  (*blocks)[num_blocks] = (long long)ftell(file);

  fclose(file);
  fprintf(stderr, "Done! Num blocks = %d, block_size=%lld, num_lines=%lld, blocks = [", curr_block, block_size, *num_lines);
  for(i=0; i<=num_blocks; i++) { fprintf(stderr, " %lld", (*blocks)[i]); }
  fprintf(stderr, " ]\n");
}

// Thang load trained word vectors
void src_LoadWordVectors(){
  FILE *fin = fopen(src_word_vector_file, "r");
  fprintf(stderr, "# Loading trained word vectors from %s ...\n", src_word_vector_file);

  // header
  fscanf(fin, "%lld %lld\n", &src_vocab_size, &layer1_size); // numWords vectorDim
  fprintf(stderr, "  num words %lld, word dim %lld\n", src_vocab_size, layer1_size);

  long a, b;
  char line[100000];
  char * pch;
  for (a = 0; a < src_vocab_size; a++) {
    fgets(line, sizeof(line), fin);
    pch = strtok (line," ");

    // read word
    strcpy(src_vocab[a].word, pch);
    pch = strtok (NULL, " ");

    // read vector
    for (b = 0; b < layer1_size; b++) {
      src_syn0[a * layer1_size + b] = atof(pch);
      pch = strtok (NULL, " ");
    }

    //    fprintf(stderr, "%s ", src_vocab[a].word);
    //    for (b = 0; b < layer1_size; b++) fprintf(stderr, "%lf ", src_syn0[a * layer1_size + b]);
    //    fprintf(stderr, "\n");

    printf("%lldK%c", (long long)a / 1000, 13);
    fflush(stdout);
  }

  fclose(fin);
}

// Thang load trained word vectors
void tgt_LoadWordVectors(){
  FILE *fin = fopen(tgt_word_vector_file, "r");
  fprintf(stderr, "# Loading trained word vectors from %s ...\n", tgt_word_vector_file);

  // header
  fscanf(fin, "%lld %lld\n", &tgt_vocab_size, &layer1_size); // numWords vectorDim
  fprintf(stderr, "  num words %lld, word dim %lld\n", tgt_vocab_size, layer1_size);

  long a, b;
  char line[100000];
  char * pch;
  for (a = 0; a < tgt_vocab_size; a++) {
    fgets(line, sizeof(line), fin);
    pch = strtok (line," ");

    // read word
    strcpy(tgt_vocab[a].word, pch);
    pch = strtok (NULL, " ");

    // read vector
    for (b = 0; b < layer1_size; b++) {
      tgt_syn0[a * layer1_size + b] = atof(pch);
      pch = strtok (NULL, " ");
    }

    //    fprintf(stderr, "%s ", tgt_vocab[a].word);
    //    for (b = 0; b < layer1_size; b++) fprintf(stderr, "%lf ", tgt_syn0[a * layer1_size + b]);
    //    fprintf(stderr, "\n");

    printf("%lldK%c", (long long)a / 1000, 13);
    fflush(stdout);
  }

  fclose(fin);
}

// Thang: factor out code from TrainModel()
void src_SaveWordVectors(char* src_output_file){
  long a, b, c, d;
  FILE *fo;
  fo = fopen(src_output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", src_vocab_size, layer1_size);
    for (a = 0; a < src_vocab_size; a++) {
      fprintf(fo, "%s ", src_vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&src_syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", src_syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    fprintf(stderr, "# Running K-means on the word vectors");
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(src_vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < src_vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < src_vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += src_syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < src_vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * src_syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < src_vocab_size; a++) fprintf(fo, "%s %d\n", src_vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

// Thang: factor out code from TrainModel()
void tgt_SaveWordVectors(char* tgt_output_file){
  long a, b, c, d;
  FILE *fo;
  fo = fopen(tgt_output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", tgt_vocab_size, layer1_size);
    for (a = 0; a < tgt_vocab_size; a++) {
      fprintf(fo, "%s ", tgt_vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&tgt_syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", tgt_syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    fprintf(stderr, "# Running K-means on the word vectors");
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(tgt_vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < tgt_vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < tgt_vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += tgt_syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < tgt_vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * tgt_syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < tgt_vocab_size; a++) fprintf(fo, "%s %d\n", tgt_vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

void execute(char* command){
  //  fprintf(stderr, "# Executing: %s\n", command);
  system(command);
}

void cldcEvaluate(char* outPrefix, int iter) {
  char command[MAX_STRING];

  /* de2en */
  // prepare data
  chdir("cldc/scripts/de2en");
  if(iter<0) { // full evaluation
    sprintf(command, "./prepare-data-klement-4cat-train-valid-my-embeddings.ch %s", outPrefix); execute(command);
    sprintf(command, "./prepare-data-klement-4cat-all-sizes-my-embeddings.ch %s", outPrefix); execute(command);
  } else {
    sprintf(command, "./prepare-data-klement-4cat-1000-my-embeddings.ch %s", outPrefix); execute(command);
  }
  // run perceptron
  if(iter<0) { // full evaluation
    sprintf(command, "./run-perceptron-train-valid-my-embeddings.ch %s", outPrefix); execute(command);
    sprintf(command, "./run-perceptron-all-sizes-my-embeddings.ch %s", outPrefix); execute(command);
    system("");
  } else {
    fprintf(stderr, "# eval %d %s %s", iter, "de2en", "cldc");
    sprintf(command, "./run-perceptron-1000-my-embeddings.ch %s", outPrefix); execute(command);
  }

  /** en2de **/
  // prepare data
  chdir("../en2de");
  if(iter<0) { // full evaluation
    sprintf(command, "./prepare-data-klement-4cat-train-valid-my-embeddings.ch %s", outPrefix); execute(command);
    sprintf(command, "./prepare-data-klement-4cat-all-sizes-my-embeddings.ch %s", outPrefix); execute(command);
  } else {
    sprintf(command, "./prepare-data-klement-4cat-1000-my-embeddings.ch %s", outPrefix); execute(command);
  }
  // run perceptron
  if(iter<0) { // full evaluation
    sprintf(command, "./run-perceptron-train-valid-my-embeddings.ch %s", outPrefix); execute(command);
    sprintf(command, "./run-perceptron-all-sizes-my-embeddings.ch %s", outPrefix); execute(command);
    system("");
  } else {
    fprintf(stderr, "# eval %d %s %s", iter, "en2de", "cldc");
    sprintf(command, "./run-perceptron-1000-my-embeddings.ch %s", outPrefix); execute(command);
  }
  chdir("../../..");
}

void evaluate(char* iter_prefix, char* srcLang, char* tgtLang, int iter) {
  char command[MAX_STRING];
  char srcWeFile[MAX_STRING], tgtWeFile[MAX_STRING];
  char srcPrefix[MAX_STRING], tgtPrefix[MAX_STRING];
  long a, b;
  FILE *file;

  fprintf(stderr, "\n");
  sprintf(srcWeFile, "%s.%s.We", output_prefix, srcLang);
  sprintf(tgtWeFile, "%s.%s.We", output_prefix, tgtLang);
  file = fopen(srcWeFile, "wb");
  for (a = 0; a < src_vocab_size; a++) {
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&src_syn0[a * layer1_size + b], sizeof(real), 1, file);
    else for (b = 0; b < layer1_size; b++) fprintf(file, "%lf ", src_syn0[a * layer1_size + b]);
    fprintf(file, "\n");
  }
  fclose(file);
  file = fopen(tgtWeFile, "wb");
  for (a = 0; a < tgt_vocab_size; a++) {
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&tgt_syn0[a * layer1_size + b], sizeof(real), 1, file);
    else for (b = 0; b < layer1_size; b++) fprintf(file, "%lf ", tgt_syn0[a * layer1_size + b]);
    fprintf(file, "\n");
  }
  fclose(file);

  sprintf(srcPrefix, "%s.%s", output_prefix, srcLang);
  sprintf(tgtPrefix, "%s.%s", output_prefix, tgtLang);

  /** WordSim **/
  chdir("wordsim/code");
  // en
  fprintf(stderr, "# eval %d %s %s", iter, srcLang, "wordSim");
  sprintf(command, "./run_wordSim.sh %s 3 %s", srcPrefix, srcLang);
  execute(command);

  fprintf(stderr, "# eval %d %s %s", iter, tgtLang, "wordSim");
  sprintf(command, "./run_wordSim.sh %s 3 %s", tgtPrefix, tgtLang); execute(command);
  chdir("../..");

  /** Analogy **/
  if((iter+1)%10==0){

    chdir("analogy/code");
    fprintf(stderr, "# eval %d %s %s", iter, "en", "analogy");
    if(strcmp(srcLang, "en")==0){
      sprintf(command, "./run_analogy.sh %s 3", srcPrefix);
    } else if(strcmp(tgtLang, "en")==0){
      sprintf(command, "./run_analogy.sh %s 3", tgtPrefix);
    }
    execute(command);
    chdir("../..");
  }

  /** CLDC **/
  // save src embs
  char srcEmbFile[MAX_STRING];
  sprintf(srcEmbFile, "%s.%s-%s.%s", iter_prefix, srcLang, tgtLang, srcLang);
  src_SaveWordVectors(srcEmbFile);

  // save tgt embs
  char tgtEmbFile[MAX_STRING];
  sprintf(tgtEmbFile, "%s.%s-%s.%s", iter_prefix, srcLang, tgtLang, tgtLang);
  tgt_SaveWordVectors(tgtEmbFile);

  if(strcmp(srcLang, "de")==0 && strcmp(tgtLang, "en")==0){
    cldcEvaluate(iter_prefix, iter);
  }
}

void print_array(int* values, int num_values){
  int i;
  for(i=0; i<num_values; i++) {
    if(i<(num_values-1)) {
      fprintf(stderr, "%d ", values[i]);
    } else {
      fprintf(stderr, "%d\n", values[i]);
    }
  }
}

int load_sent(FILE* fin, long long** sent, int side){
  char line[100000];
  char * token;

  int debug=0;
  fgets(line, sizeof(line), fin);
  // trim end of line
  int last_index=strlen(line)-1;
  while(last_index>=0 && (line[last_index]=='\n' || line[last_index]==' ' || line[last_index]=='\t')){
    line[last_index--]=0;
  }

  if(debug) { fprintf(stderr, "# Line: %s\n", line); }

  token = strtok (line," ");
  int sent_len=0;

  while(token!=NULL){ // srcId-tgtId
    // check for white spaces in front
    while(isspace(*token)) { token++; }
    if(*token==0) { // all spaces
      token = strtok (NULL, " ");
      continue;
    }

    if(debug) { fprintf(stderr, "%s\n", token); }
    if(side==0) (*sent)[sent_len] = src_SearchVocab(token);
    else (*sent)[sent_len] = tgt_SearchVocab(token);

    if(debug) { fprintf(stderr, "%d: %s, %lld\n", sent_len+1, token, (*sent)[sent_len]); }

    sent_len++;
    token = strtok (NULL, " ");
  }
  if(debug) { fprintf(stderr, "sent length = %d\n", sent_len); }

  return sent_len;
}

// Thang May14: load one alignment line of the format: srcId1 tgtId1 srcId2 tgtId2
//   return number of alignments


void print_sent(int* sent, int sent_len, struct vocab_word* vocab){
  int i;
  for(i=0; i<sent_len; i++) {
    if(i<(sent_len-1)) {
      fprintf(stderr, "%s(%d) ", vocab[sent[i]].word, i);
    } else {
      fprintf(stderr, "%s(%d)\n", vocab[sent[i]].word, i);
    }
  }
}


// last_word (input) predicts word (output).
// syn0 belongs to the input side.
// syn1neg, table, vocab_size corresponds to the output side.
void ProcessSkipPair(long long word, long long last_word, unsigned long long *next_random, int *table, real *syn0, real *syn1neg, long long vocab_size, real lambda) {
  long long d; // a, b,
  //  long long word, last_word, sentence_position;
  long long l1, l2, c, target, label;
  real f, g;
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  l1 = last_word * layer1_size; // syn0: input, last_word
  for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
  // HIERARCHICAL SOFTMAX
  if (hs) { fprintf(stderr, "hierarchical softmax is not implemented.\n"); exit(1);}

  // NEGATIVE SAMPLING
  if (negative > 0) {
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        target = word;
        label = 1;
      } else {
        (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
        target = table[((*next_random) >> 16) % table_size];
        if (target == 0) target = (*next_random) % (vocab_size - 1) + 1;
        if (target == word) continue;
        label = 0;
      }
      l2 = target * layer1_size; // syn1neg: output, word
      f = 0;
      for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
      if (f > MAX_EXP) g = (label - 1) * alpha;
      else if (f < -MAX_EXP) g = (label - 0) * alpha;
      else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
      for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
      for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += lambda * g * syn0[c + l1]; // update output embeddings
    }
    // Learn weights input -> hidden
    for (c = 0; c < layer1_size; c++) syn0[c + l1] += lambda * neu1e[c]; // update input embeddings
  }

  free(neu1e);
}

void ProcessSentenceAlign(long long* src_sent, long long src_len, int src_pos, long long* tgt_sent, long long tgt_len, int tgt_pos, unsigned long long *next_random) {
  int debug=0, neighbor_pos, offset;
  long long src_word, tgt_word, last_word;
  real range;
  src_word = src_sent[src_pos];
  tgt_word = tgt_sent[tgt_pos];

  if(debug){
    fprintf(stderr, "# generating examples for %s (%d, freq=%lld) - %s (%d, freq=%lld)\n",
        src_vocab[src_word].word, src_pos, src_vocab[src_word].cn,
        tgt_vocab[tgt_word].word, tgt_pos, tgt_vocab[tgt_word].cn);
  }
  // The subsampling randomly discards frequent words while keeping the ranking same
  if (align_sample > 0) {
    // check if we skip src word
    real ran = (sqrt(src_vocab[src_word].cn / (align_sample * src_train_words)) + 1)
                                                                                              * (align_sample * src_train_words) / src_vocab[src_word].cn;
    (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
    if (ran < ((*next_random) & 0xFFFF) / (real)65536) {
      if(debug){ fprintf(stderr, "# skip src\n"); }
      return;
    }

    // check if we skip tgt word
    ran = (sqrt(tgt_vocab[tgt_word].cn / (align_sample * tgt_train_words)) + 1)
                                                                                              * (align_sample * tgt_train_words) / tgt_vocab[tgt_word].cn;
    (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
    if (ran < ((*next_random) & 0xFFFF) / (real)65536) {
      if(debug){ fprintf(stderr, "# skip tgt\n"); }
      return;
    }
  }

  // get the range
  (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
  range = (*next_random) % window;

  // [pos - (window-range); pos + (window-range)], exclude pos
  //for (offset = range-window; offset < (window + 1 - range); ++offset) if (offset) {
  // tgt predicts src neighbors
  //neighbor_pos = src_pos + offset;
  for (offset = range; offset < window * 2 + 1 - range; ++offset) if (offset != window) {
    neighbor_pos = src_pos - window + offset;
    if (neighbor_pos >= 0 && neighbor_pos < src_len) {
      last_word = src_sent[neighbor_pos];
      if (last_word != -1) {
        ProcessSkipPair(last_word, tgt_word, next_random, src_table, tgt_syn0, src_syn1neg, src_vocab_size, 1.0);
      }
    }

    // src predicts tgt neighbors
    neighbor_pos = tgt_pos + offset;
    if (neighbor_pos >= 0 && neighbor_pos < tgt_len) {
      last_word = tgt_sent[neighbor_pos];
      if (last_word != -1) {
        ProcessSkipPair(last_word, src_word, next_random, tgt_table, src_syn0, tgt_syn1neg, tgt_vocab_size, 1.0);
      }
    }
  }
}

// side = 0 ---> src
// side = 1 ---> tgt
void ProcessSentence(long long sentence_length, long long *sen, unsigned long long *next_random, int *table, real *syn0, real *syn1neg, long long vocab_size) {
  long long a, b;
  long long word, last_word, sentence_position;
  long long c; //l1, l2, c, target, label;
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  for (sentence_position = 0; sentence_position < sentence_length; ++sentence_position) {
    word = sen[sentence_position]; // l2, syn1neg

    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
    b = (*next_random) % window;
    if (cbow) {  //train the cbow architecture
      fprintf(stderr, "cbow not implemented.\n");
      exit(1);
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c]; // l1, syn0
        if (last_word == -1) continue;
        ProcessSkipPair(word, last_word, next_random, table, syn0, syn1neg, vocab_size, monoLambda);
      } // a
    } // else skipgram
  } // sentence

  //  free(neu1);
  free(neu1e);
}



void *TrainModelThread(void *id) {
  //  fprintf(stderr, "# Thread %d, src seek to %lld, tgt seek to %lld\n", id, src_line_blocks[(long long)id], tgt_line_blocks[(long long)id]);
  long long src_sentence_length=0, tgt_sentence_length=0, word; //, last_word sentence_length = 0, sentence_position = 0;
  //long long src_parallel_word_count = 0, tgt_parallel_word_count = 0, src_parallel_last_word_count = 0, tgt_parallel_last_word_count = 0;

  // Thang Jun: use sent count to measure progress and adjust learning rate, instead of using words
  long long parallel_sent_count=0, parallel_last_sent_count=0;
  long long src_sen[MAX_SENTENCE_LENGTH + 1], tgt_sen[MAX_SENTENCE_LENGTH + 1]; //, *sen;
  long long thread_id = (long long)id;
  long long src_sentence_orig_length=0, tgt_sentence_orig_length=0;
  long long src_sen_orig[MAX_SENTENCE_LENGTH + 1], tgt_sen_orig[MAX_SENTENCE_LENGTH + 1];
  long long num_sents = 0;
  int src_pos, tgt_pos;
  char ch;
  unsigned long long next_random = thread_id;
  clock_t now;

  FILE *src_fi, *tgt_fi, *align_fi, *src_mono_fi, *tgt_mono_fi;

  // [0, mono_thread-1]: src
  // [mono_thread, 2*mono_thread-1]: tgt
  // [2*mono_thread, num_threads-1]: parallel
  /** Run Bi model **/
  if(thread_id>=2*mono_thread) {
    int block_id = thread_id - 2*mono_thread;
    fprintf(stderr, "# Thread %lld, train bi\n", thread_id);

    src_fi = fopen(src_train_file, "rb");
    tgt_fi = fopen(tgt_train_file, "rb");

    while (1) {
      fseek(src_fi, src_line_blocks[block_id], SEEK_SET);
      fseek(tgt_fi, tgt_line_blocks[block_id], SEEK_SET);

      if (useAlign) {
        align_fi = fopen(align_file, "rb");
        fseek(align_fi, align_line_blocks[block_id], SEEK_SET);
      }

      while (1) {
        // log
        if (parallel_sent_count - parallel_last_sent_count > 500) {
          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
          parallel_last_sent_count = parallel_sent_count;
          if ((debug_mode > 1)) {
            now = clock();
            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  bi", 13, alpha,
                parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1) * 100,
                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
            fflush(stdout);
            fflush(stdout);
          }
          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1));
          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        ++parallel_sent_count;

        // read src
        src_sentence_length = 0;
        src_sentence_orig_length = 0;
        while (1) {
          word = src_ReadWordIndex(src_fi);
          if (feof(src_fi)) break;
          if (word == -1) continue;
          if (word == 0) break;

          // keep the orig src
          src_sen_orig[src_sentence_orig_length] = word;
          src_sentence_orig_length++;

          // The subsampling randomly discards frequent words while keeping the ranking same
          if (sample > 0) {
            real ran = (sqrt(src_vocab[word].cn / (sample * src_train_words)) + 1) * (sample * src_train_words) / src_vocab[word].cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
          }
          src_sen[src_sentence_length] = word;
          src_sentence_length++;
          if (src_sentence_length >= MAX_SENTENCE_LENGTH) break;
        }

        // read tgt
        tgt_sentence_length = 0;
        tgt_sentence_orig_length = 0;
        while (1) {
          word = tgt_ReadWordIndex(tgt_fi);
          if (feof(tgt_fi)) break;
          if (word == -1) continue;
          if (word == 0) break;

          // keep the orig tgt
          tgt_sen_orig[tgt_sentence_orig_length] = word;
          tgt_sentence_orig_length++;

          // The subsampling randomly discards frequent words while keeping the ranking same
          if (sample > 0) {
            real ran = (sqrt(tgt_vocab[word].cn / (sample * tgt_train_words)) + 1) * (sample * tgt_train_words) / tgt_vocab[word].cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
          }
          tgt_sen[tgt_sentence_length] = word;
          tgt_sentence_length++;
          if (tgt_sentence_length >= MAX_SENTENCE_LENGTH) break;
        }

        // process alignment
        if (biTrain) {
          if (useAlign) { // using alignment
            while (fscanf(align_fi, "%d %d%c", &src_pos, &tgt_pos, &ch)) {
              ProcessSentenceAlign(src_sen_orig, src_sentence_orig_length, src_pos, tgt_sen_orig, tgt_sentence_orig_length, tgt_pos, &next_random);
              if (ch == '\n') break;
            }
          } else { // not using alignment, assuming diagonally aligned
            for (src_pos = 0; src_pos < src_sentence_orig_length; ++src_pos) {
              tgt_pos = src_pos * tgt_sentence_orig_length / src_sentence_orig_length;
              ProcessSentenceAlign(src_sen_orig, src_sentence_orig_length, src_pos, tgt_sen_orig, tgt_sentence_orig_length, tgt_pos, &next_random);
            }
          }
        }

        // process mono sentences
        ProcessSentence(src_sentence_length, src_sen, &next_random, src_table, src_syn0, src_syn1neg, src_vocab_size);
        ProcessSentence(tgt_sentence_length, tgt_sen, &next_random, tgt_table, tgt_syn0, tgt_syn1neg, tgt_vocab_size);

        ++num_sents;
        if (feof(src_fi)) break;
        if (ftell(src_fi) >= src_line_blocks[block_id + 1]) break;
      }

      if (num_sents >= threshold_per_thread) break;
    }

    if (useAlign) fclose(align_fi);
    fclose(tgt_fi);
    fclose(src_fi);
  }
  /** Run Tgt model **/
  else if(thread_id>=mono_thread) {
    int block_id = thread_id - mono_thread;
    tgt_mono_fi = fopen(tgt_train_mono, "rb");
    while (1) {
      fprintf(stderr, "# Thread %lld, train tgt, seek %lld\n", thread_id, tgt_mono_line_blocks_pos[block_id]);
      fseek(tgt_mono_fi, tgt_mono_line_blocks_pos[block_id], SEEK_SET);

      while (1) {
        // log
        if (parallel_sent_count - parallel_last_sent_count > 500) {
          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
          parallel_last_sent_count = parallel_sent_count;
          if ((debug_mode > 1)) {
            now = clock();
            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  tgt", 13, alpha,
                parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1) * 100,
                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
            fflush(stdout);
          }
          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1));
          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        ++parallel_sent_count;

        // read tgt
        tgt_sentence_length = 0;
        tgt_sentence_orig_length = 0;
        while (1) {
          word = tgt_ReadWordIndex(tgt_mono_fi);
          if (feof(tgt_mono_fi)) break;
          if (word == -1) continue;
          if (word == 0) break;

          // keep the orig tgt
          tgt_sen_orig[tgt_sentence_orig_length] = word;
          tgt_sentence_orig_length++;

          // The subsampling randomly discards frequent words while keeping the ranking same
          if (sample > 0) {
            real ran = (sqrt(tgt_vocab[word].cn / (sample * tgt_train_words)) + 1) * (sample * tgt_train_words) / tgt_vocab[word].cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
          }
          tgt_sen[tgt_sentence_length] = word;
          tgt_sentence_length++;
          if (tgt_sentence_length >= MAX_SENTENCE_LENGTH) break;
        }

        ProcessSentence(tgt_sentence_length, tgt_sen, &next_random, tgt_table, tgt_syn0, tgt_syn1neg, tgt_vocab_size);

        if (++num_sents >= threshold_per_thread) break;

        if (feof(tgt_mono_fi) || ftell(tgt_mono_fi) >= tgt_mono_line_blocks[block_id + 1]) { // reach the end of the block, go back
          tgt_mono_line_blocks_pos[block_id] = tgt_mono_line_blocks[block_id];
          break;
        }
      }

      if (num_sents >= threshold_per_thread) { // done, record position
        tgt_mono_line_blocks_pos[block_id] = ftell(tgt_mono_fi);
        if(tgt_mono_line_blocks_pos[block_id] >= tgt_mono_line_blocks[block_id + 1]){ // exceed the block, reset
          tgt_mono_line_blocks_pos[block_id] = tgt_mono_line_blocks[block_id];
        }

        break;
      }
    }

    fclose(tgt_mono_fi);
  }
  /** Run Src model **/
  else {
    int block_id = thread_id;
    src_mono_fi = fopen(src_train_mono, "rb");

    while (1) {
      fseek(src_mono_fi, src_mono_line_blocks_pos[block_id], SEEK_SET);
      fprintf(stderr, "# Thread %lld, train src, seek %lld\n", thread_id, src_mono_line_blocks_pos[block_id]);
      while (1) {
        // log
        if (parallel_sent_count - parallel_last_sent_count > 500) {
          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
          parallel_last_sent_count = parallel_sent_count;
          if ((debug_mode > 1)) {
            now = clock();
            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  src", 13, alpha,
                parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1) * 100,
                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
            fflush(stdout);
          }
          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count*num_threads/parallel_num_threads + 1));
          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        ++parallel_sent_count;

        // read src
        src_sentence_length = 0;
        src_sentence_orig_length = 0;
        while (1) {
          word = src_ReadWordIndex(src_mono_fi);
          if (feof(src_mono_fi)) break;
          if (word == -1) continue;
          if (word == 0) break;

          // keep the orig src
          src_sen_orig[src_sentence_orig_length] = word;
          src_sentence_orig_length++;

          // The subsampling randomly discards frequent words while keeping the ranking same
          if (sample > 0) {
            real ran = (sqrt(src_vocab[word].cn / (sample * src_train_words)) + 1) * (sample * src_train_words) / src_vocab[word].cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
          }
          src_sen[src_sentence_length] = word;
          src_sentence_length++;
          if (src_sentence_length >= MAX_SENTENCE_LENGTH) break;
        }

        ProcessSentence(src_sentence_length, src_sen, &next_random, src_table, src_syn0, src_syn1neg, src_vocab_size);

        if (++num_sents >= threshold_per_thread) break;
        if (feof(src_mono_fi)) break;
        if (feof(src_mono_fi) || ftell(src_mono_fi) >= src_mono_line_blocks[block_id + 1]) { // reach the end of the block, go back
          src_mono_line_blocks_pos[block_id] = src_mono_line_blocks[block_id];
          break;
        }
      }

      if (num_sents >= threshold_per_thread) { // done, record position
        src_mono_line_blocks_pos[block_id] = ftell(src_mono_fi);
        if(src_mono_line_blocks_pos[block_id] >= src_mono_line_blocks[block_id + 1]){ // exceed the block, reset
          src_mono_line_blocks_pos[block_id] = src_mono_line_blocks[block_id];
        }
        break;
      }
    }

    fclose(src_mono_fi);
  }

  pthread_exit(NULL);
  return NULL;
}

void TrainModel() {
  fprintf(stderr, "# Train model, src_train_file=%s, src_lang=%s, tgt_train_file=%s, tgt_lang=%s, monoLambda=%.2f\n",
      src_train_file, src_lang, tgt_train_file, tgt_lang, monoLambda);
  fprintf(stderr, "# src_output_file=%s, tgt_output_file=%s\n", src_output_file, tgt_output_file);
  fprintf(stderr, "# src_mono_file=%s, tgt_mono_file=%s, mono_size=%lld, src_mono_partial=%.2f, tgt_mono_partial=%.2f\n",
      src_train_mono, tgt_train_mono, mono_size, src_mono_partial, tgt_mono_partial);
  fprintf(stderr, "# anneal=%f\n", anneal);
  src_num_threads = mono_thread;
  tgt_num_threads = mono_thread;
  parallel_num_threads = num_threads-2*mono_thread;
  fprintf(stderr, "# mono_thread=%d, src_num_threads=%d, tgt_num_threads=%d, parallel_num_threads=%d\n", mono_thread,
      src_num_threads, tgt_num_threads, parallel_num_threads);

  double sum;
  real src_prob, tgt_prob, parallel_prob;
  long a;
  int i;
  char srcWordFile[MAX_STRING], tgtWordFile[MAX_STRING];
  FILE *file;

  ComputeBlockStartPoints(src_train_file, parallel_num_threads, &src_line_blocks, &parallel_line_count);
  ComputeBlockStartPoints(tgt_train_file, parallel_num_threads, &tgt_line_blocks, &parallel_line_count);
  if (useAlign) ComputeBlockStartPoints(align_file, parallel_num_threads, &align_line_blocks, &parallel_line_count);
  if (src_train_mono[0]) {
    ComputeBlockStartPoints(src_train_mono, src_num_threads, &src_mono_line_blocks, &src_mono_line_count);
    src_mono_line_blocks_pos = malloc((src_num_threads+1) * sizeof(long long));
    for(i=0; i<=src_num_threads; ++i) { src_mono_line_blocks_pos[i] = src_mono_line_blocks[i]; }
  }
  if (tgt_train_mono[0]) {
    ComputeBlockStartPoints(tgt_train_mono, tgt_num_threads, &tgt_mono_line_blocks, &tgt_mono_line_count);
    tgt_mono_line_blocks_pos = malloc((tgt_num_threads+1) * sizeof(long long));
    for(i=0; i<=tgt_num_threads; ++i) { tgt_mono_line_blocks_pos[i] = tgt_mono_line_blocks[i]; }
  }

  if (threshold_per_thread == -1) threshold_per_thread = parallel_line_count / parallel_num_threads;
  fprintf(stderr, "# parallel_line_count=%lld, src_mono_line_count=%lld, tgt_mono_line_count=%lld\n", parallel_line_count, src_mono_line_count, tgt_mono_line_count);
  fprintf(stderr, "# threshold_per_thread=%lld\n", threshold_per_thread);
  // ratio: 1/src_mono_line_count, 1/tgt_mono_line_count, 1/parallel_line_count
  // [0, leftThreshold): src
  // [leftThreshod, rightThreshod): tgt
  // [rightThreshold, 1): parallel
  sum = (double) (double) tgt_mono_line_count * parallel_line_count + (double)parallel_line_count * src_mono_line_count + src_mono_line_count * tgt_mono_line_count;
  if(sum==0) {
    left_threshold = 0;
    right_threshold = 0;
  } else {
    left_threshold = (double)parallel_line_count * tgt_mono_line_count / sum;
    right_threshold = (double)parallel_line_count * (src_mono_line_count + tgt_mono_line_count) / sum;
  }
  fprintf(stderr, "left_threshold = %.2f\tright_threshold = %.2f\n", left_threshold, right_threshold);

  if (src_read_vocab_file[0] != 0)
    src_ReadVocab(); else {
    src_LearnVocabFromTrainFile(src_train_file, parallel_line_count);
    src_parallel_train_words = src_train_words;
    if (src_train_mono[0]) {
      src_LearnVocabFromTrainFile(src_train_mono, src_mono_line_count);
      src_mono_train_words = src_train_words - src_parallel_train_words;
    }
    src_SortVocab();
    if (debug_mode > 0) {
      fprintf(stderr, "# Final src learn vocab size: %lld\n", src_vocab_size);
      fprintf(stdout, "# Final src learn vocab size: %lld\n", src_vocab_size);
    }
  }
  fprintf(stderr, "# src_parallel_train_words=%lld, src_mono_train_words=%lld, src_train_words=%lld\n",
      src_parallel_train_words, src_mono_train_words, src_train_words);

  if (src_save_vocab_file[0] != 0) src_SaveVocab();
  if (src_output_file[0] == 0) return;
  sprintf(srcWordFile, "%s.%s.words", output_prefix, src_lang);
  file = fopen(srcWordFile, "wb");
  for (a = 0; a < src_vocab_size; ++a) {
    fprintf(file, "%s\n", src_vocab[a].word);
  }
  fclose(file);

  if (tgt_read_vocab_file[0] != 0) tgt_ReadVocab(); else {
    tgt_LearnVocabFromTrainFile(tgt_train_file, parallel_line_count);
    tgt_parallel_train_words = tgt_train_words;
    if (tgt_train_mono[0]) {
      tgt_LearnVocabFromTrainFile(tgt_train_mono, tgt_mono_line_count);
      tgt_mono_train_words = tgt_train_words - tgt_parallel_train_words;
    }
    tgt_SortVocab();
    if (debug_mode > 0) {
      fprintf(stderr, "# Final tgt learn vocab size: %lld\n", tgt_vocab_size);
      fprintf(stdout, "# Final tgt learn vocab size: %lld\n", tgt_vocab_size);
    }
  }
  fprintf(stderr, "# tgt_parallel_train_words=%lld, tgt_mono_train_words=%lld, tgt_train_words=%lld\n",
        tgt_parallel_train_words, tgt_mono_train_words, tgt_train_words);
  if (tgt_save_vocab_file[0] != 0) tgt_SaveVocab();
  if (tgt_output_file[0] == 0) return;
  sprintf(tgtWordFile, "%s.%s.words", output_prefix, tgt_lang);
  file = fopen(tgtWordFile, "wb");
  for (a = 0; a < tgt_vocab_size; ++a) {
    fprintf(file, "%s\n", tgt_vocab[a].word);
  }
  fclose(file);

  InitNet();

  // Thang load trained word vectors
  if (is_src_word_vector_file) src_LoadWordVectors();
  if (is_tgt_word_vector_file) tgt_LoadWordVectors();

  if (src_is_train==1 || tgt_is_train==1){
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    fprintf(stderr, "# Starting training using src-file %s, tgt-file %s, and align-file %s\n", src_train_file, tgt_train_file, align_file);

    if (negative > 0) {
      src_InitUnigramTable();
      tgt_InitUnigramTable();
    }

    start = clock();
    real orig_alpha = alpha;
    for(cur_iter=start_iter; cur_iter<num_train_iters; cur_iter++){
      starting_alpha = orig_alpha; // Thang: reset learning rate for every iteration
      parallel_sent_count_actual = 0;

      fprintf(stderr, "# Start iter %d, num_threads=%d\n", cur_iter, num_threads);
      fprintf(stderr, "  left_threshold = %.2f\tright_threshold = %.2f\n", left_threshold, right_threshold);

      for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
      for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

      if (logEvery > 0 && (cur_iter+1) % logEvery == 0) {
        char iter_output_prefix[1000];
        sprintf(iter_output_prefix, "%s.%d", output_prefix, cur_iter);
        evaluate(iter_output_prefix, src_lang, tgt_lang, cur_iter);
      }

      // Thang Jun: update left/rightThreshold
      if(anneal!=1){
        src_prob = anneal*left_threshold;
        tgt_prob = anneal*(right_threshold-left_threshold);
        parallel_prob = 1 - right_threshold;
        sum = src_prob + tgt_prob + parallel_prob;
        left_threshold = src_prob/sum;
        right_threshold = (src_prob+tgt_prob)/sum;
      }
    }
    if (logEvery <= 0) {
      char iter_output_prefix[1000];
      sprintf(iter_output_prefix, "%s.%d", output_prefix, cur_iter);
      evaluate(iter_output_prefix, src_lang, tgt_lang, cur_iter);
    }
  } else {
    src_SaveWordVectors(src_output_file);
    tgt_SaveWordVectors(tgt_output_file);
  }

  free(src_line_blocks);
  free(tgt_line_blocks);
  if (useAlign) free(align_line_blocks);
  if (src_train_mono[0]) { free(src_mono_line_blocks); free(src_mono_line_blocks_pos); }
  if (tgt_train_mono[0]) { free(tgt_mono_line_blocks); free(tgt_mono_line_blocks_pos); }
}


int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  srand(0);
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");

    // Thang: to load trained vectors
    printf("\t-vector <file>\n");
    printf("\t\tUse <file> to load trained word vectors in text format provided by word2vec\n");
    // Thang: add number of training iterations
    printf("\t-iter <int>\n");
    printf("\t\tThe number of training iterations\n");
    printf("\t-startIter <int>\n"); // the iteration to start from (default 0)
    printf("\t\tThe iteration to start from (default 0)\n");
    printf("\t-anneal <float>\n"); // the iteration to start from (default 0)
        printf("\t\tAnnealing rate to decrease the importance of bilingual component (default 1)\n");

    // Hieu: evaluate after logEvery iterations
    printf("\t-logEvery <int>\n"); // the iteration to start from (default 0)
    printf("\t\tEvaluate after logEvery iteration, -1 to evaluate only in the end\n");
    printf("\t-monoLambda <int>\n"); // the monoLambda (default 1.0)
    printf("\t\tmonoLambda to anneal the effect of mono lingual model (default 1.0)\n");
    printf("\t-threshold_per_thread <int>\n"); // number of sentences we are willing to read per thread (default 100k)
    printf("\t\tnumber of sentences we are willing to read per thread (default 100k)\n");
    printf("\t-src-mono-partial <int>\n"); // ratio of src mono we wish to use
    printf("\t\tratio of src mono data we wish to use (default 1.0)\n");
    printf("\t-tgt-mono-partial <int>\n"); // ratio of src mono we wish to use
    printf("\t\tratio of tgt mono data we wish to use (default 1.0)\n");

    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }

  src_save_vocab_file[0] = 0;
  src_read_vocab_file[0] = 0;
  tgt_save_vocab_file[0] = 0;
  tgt_read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-src-train", argc, argv)) > 0) strcpy(src_train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-train", argc, argv)) > 0) strcpy(tgt_train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-align", argc, argv)) > 0) { strcpy(align_file, argv[i + 1]); useAlign = 1; }
  if ((i = ArgPos((char *)"-src-train-mono", argc, argv)) > 0) strcpy(src_train_mono, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-train-mono", argc, argv)) > 0) strcpy(tgt_train_mono, argv[i + 1]);
  if ((i = ArgPos((char *)"-src-lang", argc, argv)) > 0) strcpy(src_lang, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-lang", argc, argv)) > 0) strcpy(tgt_lang, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_prefix, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-align-sample", argc, argv)) > 0) align_sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-biTrain", argc, argv)) > 0) biTrain = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-logEvery", argc, argv)) > 0) logEvery = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-monoLambda", argc, argv)) > 0) monoLambda = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold-per-thread", argc, argv)) > 0) threshold_per_thread = atoll(argv[i + 1]);
  if ((i = ArgPos((char *)"-src-mono-partial", argc, argv)) > 0) src_mono_partial = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-mono-partial", argc, argv)) > 0) tgt_mono_partial = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-size", argc, argv)) > 0) mono_size = atoll(argv[i + 1]); // number of sentences used for each mono corpus, override src/tgt_mono_partial
  if ((i = ArgPos((char *)"-mono-thread", argc, argv)) > 0) mono_thread = atoi(argv[i + 1]); // number of sentences used for each mono corpus, override src/tgt_mono_partial

  if ((i = ArgPos((char *)"-src-save-vocab", argc, argv)) > 0) strcpy(src_save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-src-read-vocab", argc, argv)) > 0) strcpy(src_read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-save-vocab", argc, argv)) > 0) strcpy(tgt_save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-read-vocab", argc, argv)) > 0) strcpy(tgt_read_vocab_file, argv[i + 1]);

  // Thang -vector option
  if ((i = ArgPos((char *)"-src-vector", argc, argv)) > 0) {
    strcpy(src_word_vector_file, argv[i + 1]);
    is_src_word_vector_file = 1;
    if (classes>0) src_is_train = 0; // we want to compute classes from the existing vectors
  }

  if ((i = ArgPos((char *)"-tgt-vector", argc, argv)) > 0) {
    strcpy(tgt_word_vector_file, argv[i + 1]);
    is_tgt_word_vector_file = 1;
    if (classes>0) tgt_is_train = 0; // we want to compute classes from the existing vectors
  }

  // Thang -iter option
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) num_train_iters = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-startIter", argc, argv)) > 0) start_iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-anneal", argc, argv)) > 0) anneal = atof(argv[i + 1]);

  char actual_path [MAX_STRING];
  realpath(output_prefix, actual_path);
  strcpy(output_prefix, actual_path);
  sprintf(src_output_file, "%s.%s", output_prefix, src_lang);
  sprintf(tgt_output_file, "%s.%s", output_prefix, tgt_lang);

  src_vocab = (struct vocab_word *)calloc(src_vocab_max_size, sizeof(struct vocab_word));
  src_vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  tgt_vocab = (struct vocab_word *)calloc(tgt_vocab_max_size, sizeof(struct vocab_word));
  tgt_vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  //  evaluate(output_prefix, src_lang, tgt_lang, 1);

  //  // src
  //  FILE* src_fin = fopen(src_train_file, "r");
  //  int *src_sent = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int src_len = load_sent(src_fin, &src_sent, 1);
  //
  //  // tgt
  //  FILE* tgt_fin = fopen(tgt_train_file, "r");
  //  int *tgt_sent = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int tgt_len = load_sent(tgt_fin, &tgt_sent, 0);
  //
  //  // align
  //  FILE* align_fin = fopen(align_file, "r");
  //  int *src_aligns = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int *tgt_aligns = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int num_aligns = load_sent_align(align_fin, &src_aligns, &tgt_aligns);
  //
  //  // get align examples
  //  int *src_examples = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int *tgt_examples = malloc(MAX_SENT_LENGTH * sizeof(int));
  //  int num_examples = build_align_examples(src_aligns, tgt_aligns, num_aligns,
  //      src_sent, src_len, tgt_sent, tgt_len, &src_examples, &tgt_examples);

  return 0;
}


//void *TrainModelThreadOld(void *id) {
//  //  fprintf(stderr, "# Thread %d, src seek to %lld, tgt seek to %lld\n", id, src_line_blocks[(long long)id], tgt_line_blocks[(long long)id]);
//  long long src_sentence_length=0, tgt_sentence_length=0, word; //, last_word sentence_length = 0, sentence_position = 0;
//  //long long src_parallel_word_count = 0, tgt_parallel_word_count = 0, src_parallel_last_word_count = 0, tgt_parallel_last_word_count = 0;
//
//  // Thang Jun: use sent count to measure progress and adjust learning rate, instead of using words
//  long long parallel_sent_count=0, parallel_last_sent_count=0;
//  long long src_sen[MAX_SENTENCE_LENGTH + 1], tgt_sen[MAX_SENTENCE_LENGTH + 1]; //, *sen;
//  long long thread_id = (long long)id;
//  long long src_sentence_orig_length=0, tgt_sentence_orig_length=0;
//  long long src_sen_orig[MAX_SENTENCE_LENGTH + 1], tgt_sen_orig[MAX_SENTENCE_LENGTH + 1];
//  long long num_sents = 0;
//  int src_pos, tgt_pos;
//  char ch;
//  unsigned long long next_random = thread_id;
//  clock_t now;
//
//  FILE *src_fi, *tgt_fi, *align_fi, *src_mono_fi, *tgt_mono_fi;
//
//  real decider = (real)rand() / RAND_MAX;
//  if (decider < 0) decider = -decider;
//  /** Run Bi model **/
//  if (decider >= right_threshold) {
//    //fprintf(stderr, "# Thread %lld, train bi, decider=%f\n", (long long) id, decider);
//    src_fi = fopen(src_train_file, "rb");
//    tgt_fi = fopen(tgt_train_file, "rb");
//
//    while (1) {
//      fseek(src_fi, src_line_blocks[thread_id], SEEK_SET);
//      fseek(tgt_fi, tgt_line_blocks[thread_id], SEEK_SET);
//      //fprintf(stderr, "# Thread %d, seek to src %lld, tgt %lld\n", id, src_line_blocks[thread_id], tgt_line_blocks[thread_id]);
//      if (useAlign) {
//        //fprintf(stderr, "# Thread %d, align seek to %lld\n", id, align_line_blocks[thread_id]);
//        align_fi = fopen(align_file, "rb");
//        fseek(align_fi, align_line_blocks[thread_id], SEEK_SET);
//      }
//
//      while (1) {
//        // log
//        if (parallel_sent_count - parallel_last_sent_count > 500) {
//          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
//          parallel_last_sent_count = parallel_sent_count;
//          if ((debug_mode > 1)) {
//            now = clock();
//            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  bi", 13, alpha,
//            //printf("Thread %d, actual_count=%lld, parallel_line_count=%lld, Alpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk bi\n",
////                (long long) id, parallel_sent_count_actual, parallel_line_count, alpha,
//                parallel_sent_count_actual / (real)(parallel_line_count + 1) * 100,
//                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
//            fflush(stdout);
//            fflush(stdout);
//          }
//          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count + 1));
//          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
//        }
//        ++parallel_sent_count;
//
//        // read src
//        src_sentence_length = 0;
//        src_sentence_orig_length = 0;
//        while (1) {
//          word = src_ReadWordIndex(src_fi);
//          if (feof(src_fi)) break;
//          if (word == -1) continue;
//          if (word == 0) break;
//
//          // keep the orig src
//          src_sen_orig[src_sentence_orig_length] = word;
//          src_sentence_orig_length++;
//
//          // The subsampling randomly discards frequent words while keeping the ranking same
//          if (sample > 0) {
//            real ran = (sqrt(src_vocab[word].cn / (sample * src_train_words)) + 1) * (sample * src_train_words) / src_vocab[word].cn;
//            next_random = next_random * (unsigned long long)25214903917 + 11;
//            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
//          }
//          src_sen[src_sentence_length] = word;
//          src_sentence_length++;
//          if (src_sentence_length >= MAX_SENTENCE_LENGTH) break;
//        }
//
//        // read tgt
//        tgt_sentence_length = 0;
//        tgt_sentence_orig_length = 0;
//        while (1) {
//          word = tgt_ReadWordIndex(tgt_fi);
//          if (feof(tgt_fi)) break;
//          if (word == -1) continue;
//          if (word == 0) break;
//
//          // keep the orig tgt
//          tgt_sen_orig[tgt_sentence_orig_length] = word;
//          tgt_sentence_orig_length++;
//
//          // The subsampling randomly discards frequent words while keeping the ranking same
//          if (sample > 0) {
//            real ran = (sqrt(tgt_vocab[word].cn / (sample * tgt_train_words)) + 1) * (sample * tgt_train_words) / tgt_vocab[word].cn;
//            next_random = next_random * (unsigned long long)25214903917 + 11;
//            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
//          }
//          tgt_sen[tgt_sentence_length] = word;
//          tgt_sentence_length++;
//          if (tgt_sentence_length >= MAX_SENTENCE_LENGTH) break;
//        }
//
//        // process alignment
//        if (biTrain) {
//          if (useAlign) { // using alignment
//            while (fscanf(align_fi, "%d %d%c", &src_pos, &tgt_pos, &ch)) {
//              ProcessSentenceAlign(src_sen_orig, src_sentence_orig_length, src_pos, tgt_sen_orig, tgt_sentence_orig_length, tgt_pos, &next_random);
//              if (ch == '\n') break;
//            }
//          } else { // not using alignment, assuming diagonally aligned
//            for (src_pos = 0; src_pos < src_sentence_orig_length; ++src_pos) {
//              tgt_pos = src_pos * tgt_sentence_orig_length / src_sentence_orig_length;
//              ProcessSentenceAlign(src_sen_orig, src_sentence_orig_length, src_pos, tgt_sen_orig, tgt_sentence_orig_length, tgt_pos, &next_random);
//            }
//          }
//        }
//
//        // process mono sentences
//        ProcessSentence(src_sentence_length, src_sen, &next_random, src_table, src_syn0, src_syn1neg, src_vocab_size);
//        ProcessSentence(tgt_sentence_length, tgt_sen, &next_random, tgt_table, tgt_syn0, tgt_syn1neg, tgt_vocab_size);
//
//        ++num_sents;
//        if (feof(src_fi)) break;
//        if (ftell(src_fi) >= src_line_blocks[thread_id + 1]) break;
//      }
//
//      if (num_sents >= threshold_per_thread) break;
//    }
//
//    if (useAlign) fclose(align_fi);
//    fclose(tgt_fi);
//    fclose(src_fi);
//  }
//  /** Run Tgt model **/
//  else if (decider >= left_threshold) {
//    //fprintf(stderr, "# Thread %lld, train tgt, decider=%f\n", (long long) id, decider);
//    tgt_mono_fi = fopen(tgt_train_mono, "rb");
//    while (1) {
//      fseek(tgt_mono_fi, tgt_mono_line_blocks_pos[thread_id], SEEK_SET);
//
//      while (1) {
//        // log
//        if (parallel_sent_count - parallel_last_sent_count > 500) {
//          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
//          parallel_last_sent_count = parallel_sent_count;
//          if ((debug_mode > 1)) {
//            now = clock();
//            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  tgt", 13, alpha,
////            printf("Thread %d, actual_count=%lld, parallel_line_count=%lld, Alpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk tgt\n",
////                (long long) id, parallel_sent_count_actual, parallel_line_count, alpha,
//                parallel_sent_count_actual / (real)(parallel_line_count + 1) * 100,
//                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
//            fflush(stdout);
//          }
//          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count + 1));
//          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
//        }
//        ++parallel_sent_count;
//
//        // read tgt
//        tgt_sentence_length = 0;
//        tgt_sentence_orig_length = 0;
//        while (1) {
//          word = tgt_ReadWordIndex(tgt_mono_fi);
//          if (feof(tgt_mono_fi)) break;
//          if (word == -1) continue;
//          if (word == 0) break;
//
//          // keep the orig tgt
//          tgt_sen_orig[tgt_sentence_orig_length] = word;
//          tgt_sentence_orig_length++;
//
//          // The subsampling randomly discards frequent words while keeping the ranking same
//          if (sample > 0) {
//            real ran = (sqrt(tgt_vocab[word].cn / (sample * tgt_train_words)) + 1) * (sample * tgt_train_words) / tgt_vocab[word].cn;
//            next_random = next_random * (unsigned long long)25214903917 + 11;
//            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
//          }
//          tgt_sen[tgt_sentence_length] = word;
//          tgt_sentence_length++;
//          if (tgt_sentence_length >= MAX_SENTENCE_LENGTH) break;
//        }
//
//        ProcessSentence(tgt_sentence_length, tgt_sen, &next_random, tgt_table, tgt_syn0, tgt_syn1neg, tgt_vocab_size);
//
//        if (++num_sents >= threshold_per_thread) break;
//
//        if (feof(tgt_mono_fi) || ftell(tgt_mono_fi) >= tgt_mono_line_blocks[thread_id + 1]) { // reach the end of the block, go back
//          tgt_mono_line_blocks_pos[thread_id] = tgt_mono_line_blocks[thread_id];
//          // fseek(tgt_mono_fi, tgt_mono_line_blocks[thread_id], SEEK_SET);
//          break;
//        }
//      }
//
//      if (num_sents >= threshold_per_thread) { // done, record position
//        tgt_mono_line_blocks_pos[thread_id] = ftell(tgt_mono_fi);
//        break;
//      }
//    }
//
//    fclose(tgt_mono_fi);
//  }
//  /** Run Src model **/
//  else {
//    //fprintf(stderr, "# Thread %lld, train src, decider=%f\n", (long long) id, decider);
//    src_mono_fi = fopen(src_train_mono, "rb");
//
//    while (1) {
//      fseek(src_mono_fi, src_mono_line_blocks_pos[thread_id], SEEK_SET);
//
//      while (1) {
//        // log
//        if (parallel_sent_count - parallel_last_sent_count > 500) {
//          parallel_sent_count_actual += parallel_sent_count - parallel_last_sent_count;
//          parallel_last_sent_count = parallel_sent_count;
//          if ((debug_mode > 1)) {
//            now = clock();
//            printf("%cAlpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  src", 13, alpha,
////            printf("Thread %d, actual_count=%lld, parallel_line_count=%lld, Alpha: %f  Progress: %.2f%%  Sents/thread/sec: %.2fk  src\n",
////                (long long) id, parallel_sent_count_actual, parallel_line_count, alpha,
//                parallel_sent_count_actual / (real)(parallel_line_count + 1) * 100,
//                parallel_sent_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
//            fflush(stdout);
//          }
//          alpha = starting_alpha * (1 - parallel_sent_count_actual / (real)(parallel_line_count + 1));
//          if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
//        }
//        ++parallel_sent_count;
//
//        // read src
//        src_sentence_length = 0;
//        src_sentence_orig_length = 0;
//        while (1) {
//          word = src_ReadWordIndex(src_mono_fi);
//          if (feof(src_mono_fi)) break;
//          if (word == -1) continue;
//          if (word == 0) break;
//
//          // keep the orig src
//          src_sen_orig[src_sentence_orig_length] = word;
//          src_sentence_orig_length++;
//
//          // The subsampling randomly discards frequent words while keeping the ranking same
//          if (sample > 0) {
//            real ran = (sqrt(src_vocab[word].cn / (sample * src_train_words)) + 1) * (sample * src_train_words) / src_vocab[word].cn;
//            next_random = next_random * (unsigned long long)25214903917 + 11;
//            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
//          }
//          src_sen[src_sentence_length] = word;
//          src_sentence_length++;
//          if (src_sentence_length >= MAX_SENTENCE_LENGTH) break;
//        }
//
//        ProcessSentence(src_sentence_length, src_sen, &next_random, src_table, src_syn0, src_syn1neg, src_vocab_size);
//
//        if (++num_sents >= threshold_per_thread) break;
//        if (feof(src_mono_fi)) break;
//        if (feof(src_mono_fi) || ftell(src_mono_fi) >= src_mono_line_blocks[thread_id + 1]) { // reach the end of the block, go back
//          src_mono_line_blocks_pos[thread_id] = src_mono_line_blocks[thread_id];
//          // fseek(src_mono_fi, src_mono_line_blocks[thread_id], SEEK_SET);
//          break;
//        }
//      }
//
//      if (num_sents >= threshold_per_thread) { // stop, record where we were
//        src_mono_line_blocks_pos[thread_id] = ftell(src_mono_fi);
//        break;
//      }
//    }
//
//    fclose(src_mono_fi);
//  }
//  //fprintf(stderr, "# Thread %lld stop after processing %lld sents\n", (long long) id, num_sents);
//  pthread_exit(NULL);
//  return NULL;
//}
