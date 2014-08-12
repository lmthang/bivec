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
#include <float.h>
#include <unistd.h>
#include <assert.h>

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_STRING_LENGTH 10000
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct train_params {
  char lang[MAX_STRING];
  char train_file[MAX_STRING];
  char output_file[MAX_STRING];
  char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
  struct vocab_word *vocab;
  int *vocab_hash;
  long long train_words, word_count_actual, file_size;

  // syn0 belongs to the input side.
  // syn1neg, table, vocab_size corresponds to the output side.
  long long vocab_max_size, vocab_size;
  real *syn0, *syn1, *syn1neg;
  int *table;

  // line blocks
  long long num_lines;
  long long *line_blocks;
};

int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
long long layer1_size = 100;
long long classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;

// Thang and Hieu, Jul 2014
struct train_params *src;
int eval_opt = -1; // evaluation option
int num_train_iters = 1, cur_iter = 0, start_iter = 0; // run multiple iterations
char output_prefix[MAX_STRING]; // output_prefix.lang: stores embeddings
char align_file[MAX_STRING];

// tgt
int is_tgt = 0;
struct train_params *tgt;

// align
int use_align = 0;
long long align_num_lines;
long long *align_line_blocks;

void InitUnigramTable(struct train_params *params) {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  int *table = params->table;
  long long vocab_size = params->vocab_size;
  struct vocab_word *vocab = params->vocab;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
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
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word, const struct vocab_word *vocab, const int *vocab_hash) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) {
      return vocab_hash[hash];
    }
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, const struct vocab_word *vocab, const int *vocab_hash) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word, vocab, vocab_hash);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, struct train_params *params) {
  unsigned int hash, length = strlen(word) + 1;
  long long vocab_size = params->vocab_size;
  long long vocab_max_size = params->vocab_max_size;
  struct vocab_word *vocab = params->vocab;
  int *vocab_hash = params->vocab_hash;

  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
//  printf("map %d -> %d\n", hash);
  params->vocab_size = vocab_size;
  params->vocab_max_size = vocab_max_size;
  params->vocab = vocab;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct train_params *params) {
  int a, size;
  unsigned int hash;
  int *vocab_hash = params->vocab_hash;
  struct vocab_word *vocab = params->vocab;
  long long vocab_size = params->vocab_size;

  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  params->train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      params->train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }

  params->vocab = vocab;
  params->vocab_size = vocab_size;
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct train_params *params) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < params->vocab_size; a++) if (params->vocab[a].cn > min_reduce) {
    params->vocab[b].cn = params->vocab[a].cn;
    params->vocab[b].word = params->vocab[a].word;
    b++;
  } else free(params->vocab[a].word);
  params->vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  for (a = 0; a < params->vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(params->vocab[a].word);
    while (params->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    params->vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(struct train_params *params) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < params->vocab_size; a++) count[a] = params->vocab[a].cn;
  for (a = params->vocab_size; a < params->vocab_size * 2; a++) count[a] = 1e15;
  pos1 = params->vocab_size - 1;
  pos2 = params->vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < params->vocab_size - 1; a++) {
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
    count[params->vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = params->vocab_size + a;
    parent_node[min2i] = params->vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < params->vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == params->vocab_size * 2 - 2) break;
    }
    params->vocab[a].codelen = i;
    params->vocab[a].point[0] = params->vocab_size - 2;
    for (b = 0; b < i; b++) {
      params->vocab[a].code[i - b - 1] = code[b];
      params->vocab[a].point[i - b] = point[b] - params->vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile(struct train_params *params) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  fin = fopen(params->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  params->vocab_size = 0;
  AddWordToVocab((char *)"</s>", params);
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    params->train_words++;
    if ((debug_mode > 1) && (params->train_words % 100000 == 0)) {
      printf("%lldK%c", params->train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word, params->vocab, params->vocab_hash);

    if (i == -1) {
      a = AddWordToVocab(word, params);
      params->vocab[a].cn = 1;
    } else params->vocab[i].cn++;
    if (params->vocab_size > vocab_hash_size * 0.7) ReduceVocab(params);
  }
  SortVocab(params);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", params->vocab_size);
    printf("Words in train file: %lld\n", params->train_words);
  }
  params->file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab(struct train_params *params) {
  long long i;
  FILE *fo = fopen(params->save_vocab_file, "wb");
  for (i = 0; i < params->vocab_size; i++) fprintf(fo, "%s %lld\n", params->vocab[i].word, params->vocab[i].cn);
  fclose(fo);
}

void ReadVocab(struct train_params *params) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(params->read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  params->vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word, params);
    fscanf(fin, "%lld%c", &params->vocab[a].cn, &c);
    i++;
  }
  SortVocab(params);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", params->vocab_size);
    printf("Words in train file: %lld\n", params->train_words);
  }
  fin = fopen(params->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  params->file_size = ftell(fin);
  fclose(fin);
}

void InitNet(struct train_params *params) {
  long long a, b;
  a = posix_memalign((void **)&params->syn0, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
  if (params->syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    printf("# Init hierarchical softmax syn1: %lld x %lld\n", params->vocab_size, layer1_size);
    a = posix_memalign((void **)&params->syn1, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
     params->syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    printf("# Init negative sampling syn1neg: %lld x %lld\n", params->vocab_size, layer1_size);
    a = posix_memalign((void **)&params->syn1neg, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
     params->syn1neg[a * layer1_size + b] = 0;
  }
  for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
  params->syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  CreateBinaryTree(params);
}

void stat(real* a_syn, long long num_elements, char* name){
  float min = FLT_MAX;
  float max = FLT_MIN;
  float avg = 0;
  long long i;
  for(i=0; i<num_elements; ++i){
    if (a_syn[i]>max) max = a_syn[i];
    if (a_syn[i]<min) min = a_syn[i];
    avg += a_syn[i];
  }
  avg /= num_elements;
  printf("%s: min=%f, max=%f, avg=%f\n", name, min, max, avg);
}

// long long read_sentence(FILE *fi, struct train_params *params, long long *sen, unsigned long long *next_random) {
//   long long word, sentence_length = 0;
//   while (1) {
//     word = ReadWordIndex(fi, params->vocab, params->vocab_hash);
//     if (feof(fi)) break;
//     if (word == -1) continue;
//     if (word == 0) break;
//     // The subsampling randomly discards frequent words while keeping the ranking same
//     if (sample > 0) {
//       real ran = (sqrt(params->vocab[word].cn / (sample * params->train_words)) + 1) * (sample * params->train_words) / params->vocab[word].cn;
//       *next_random = *next_random * (unsigned long long)25214903917 + 11;
//       if (ran < (*next_random & 0xFFFF) / (real)65536) continue;
//     }
//     sen[sentence_length++] = word;
//     if (sentence_length >= MAX_SENTENCE_LENGTH) break;
//   }
//   return sentence_length;
// }

void ComputeBlockStartPoints(char* file_name, int num_blocks, long long **blocks, long long *num_lines) {
  printf("# ComputeBlockStartPoints %s, num_blocks=%d\n", file_name, num_blocks);
  long long block_size;
  int line_count = 0;
  int curr_block = 0;
  char line[MAX_STRING_LENGTH];
  FILE *file;

  *num_lines = 0;
  file = fopen(file_name, "r");
  while (1) {
    fgets(line, MAX_STRING_LENGTH, file);
    if (feof(file)) break;
    ++(*num_lines);
  }
  printf("  num_lines=%lld\n", *num_lines);

//  if (strcmp(file_name, src_train_mono)==0) {
//    if(mono_size>=0 && mono_size<(*num_lines)){ // use specific size
//      *num_lines = mono_size;
//    } else { // use proportion
//      *num_lines = (*num_lines) * src_mono_partial;
//    }
//  }
//  if (strcmp(file_name, tgt_train_mono)==0) {
//    if(mono_size>=0 && mono_size<(*num_lines)){ // use specific size
//      *num_lines = mono_size;
//    } else { // use proportion
//      *num_lines = (*num_lines) * tgt_mono_partial;
//    }
//  }

  fseek(file, 0, SEEK_SET);
  block_size = (*num_lines - 1) / num_blocks + 1;
  printf("  block_size=%lld\n", block_size);

  *blocks = malloc((num_blocks+1) * sizeof(long long));
  (*blocks)[0] = 0;
  curr_block = 0;
  long long int cur_size = 0;
  while (1) {
    fgets(line, MAX_STRING_LENGTH, file);
    line_count++;
    cur_size++;

    // done with a block or reach eof
    if (cur_size == block_size || line_count==(*num_lines)) {
      curr_block++;
      (*blocks)[curr_block] = (long long)ftell(file);
      printf("  block[%d]=%lld\n", curr_block, (*blocks)[curr_block]);
      if (line_count==(*num_lines)) { // eof
        break;
      }

      // reset
      cur_size = 0;
    }
  }
  assert(curr_block==num_blocks);
  assert(line_count==(*num_lines));

  fclose(file);
}

void ProcessCbow(long long word, unsigned long long *next_random, struct train_params *src, real *neu1, real *neu1e) {
  long long d;
  long long l2, c, target, label;
  real f, g;

  if (hs) for (d = 0; d < src->vocab[word].codelen; d++) {
    f = 0;
    l2 = src->vocab[word].point[d] * layer1_size;
    // Propagate hidden -> output
    for (c = 0; c < layer1_size; c++) f += neu1[c] * src->syn1[c + l2];
    if (f <= -MAX_EXP) continue;
    else if (f >= MAX_EXP) continue;
    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    // 'g' is the gradient multiplied by the learning rate
    g = (1 - src->vocab[word].code[d] - f) * alpha;
    // Propagate errors output -> hidden
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * src->syn1[c + l2];
    // Learn weights hidden -> output
    for (c = 0; c < layer1_size; c++) src->syn1[c + l2] += g * neu1[c];
  }
  // NEGATIVE SAMPLING
  if (negative > 0) for (d = 0; d < negative + 1; d++) {
    if (d == 0) {
      target = word;
      label = 1;
    } else {
      *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
      target = src->table[((*next_random) >> 16) % table_size];
      if (target == 0) target = (*next_random) % (src->vocab_size - 1) + 1;
      if (target == word) continue;
      label = 0;
    }
    l2 = target * layer1_size;
    f = 0;
    for (c = 0; c < layer1_size; c++) f += neu1[c] * src->syn1neg[c + l2];
    if (f > MAX_EXP) g = (label - 1) * alpha;
    else if (f < -MAX_EXP) g = (label - 0) * alpha;
    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * src->syn1neg[c + l2];
    for (c = 0; c < layer1_size; c++) src->syn1neg[c + l2] += g * neu1[c];
  }
}

// last_word (input) predicts word (output).
// syn0 belongs to the input side.
// syn1neg, table, vocab_size corresponds to the output side.
void ProcessSkipPair(long long word, long long last_word, unsigned long long *next_random, struct train_params *src, real *neu1, real *neu1e) {
  long long d;
  long long l1, l2, c, target, label;
  real f, g;

  l1 = last_word * layer1_size;
  for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
  // HIERARCHICAL SOFTMAX
  if (hs) for (d = 0; d < src->vocab[word].codelen; d++) {
    f = 0;
    l2 = src->vocab[word].point[d] * layer1_size;
    // Propagate hidden -> output
    for (c = 0; c < layer1_size; c++) f += src->syn0[c + l1] * src->syn1[c + l2];
    if (f <= -MAX_EXP) continue;
    else if (f >= MAX_EXP) continue;
    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    // 'g' is the gradient multiplied by the learning rate
    g = (1 - src->vocab[word].code[d] - f) * alpha;
    // Propagate errors output -> hidden
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * src->syn1[c + l2];
    // Learn weights hidden -> output
    for (c = 0; c < layer1_size; c++) src->syn1[c + l2] += g * src->syn0[c + l1];
  }
  // NEGATIVE SAMPLING
  if (negative > 0) for (d = 0; d < negative + 1; d++) {
    if (d == 0) {
      target = word;
      label = 1;
    } else {
      *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
      target = src->table[((*next_random) >> 16) % table_size];
      if (target == 0) target = (*next_random) % (src->vocab_size - 1) + 1;
      if (target == word) continue;
      label = 0;
    }
    l2 = target * layer1_size;
    f = 0;
    for (c = 0; c < layer1_size; c++) f += src->syn0[c + l1] * src->syn1neg[c + l2];
    if (f > MAX_EXP) g = (label - 1) * alpha;
    else if (f < -MAX_EXP) g = (label - 0) * alpha;
    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * src->syn1neg[c + l2];
    for (c = 0; c < layer1_size; c++) src->syn1neg[c + l2] += g * src->syn0[c + l1];
  }
  // Learn weights input -> hidden
  for (c = 0; c < layer1_size; c++) src->syn0[c + l1] += neu1e[c];

}

// side = 0 ---> src
// side = 1 ---> tgt
// neu1: cbow
// neu1e: skipgram
// syn0: input embeddings (both hs and negative)
// syn1: output embeddings (hs)
// syn1neg: output embeddings (negative)
void ProcessSentence(long long sentence_length, long long *sen, struct train_params *src, unsigned long long *next_random, real *neu1, real *neu1e) {
  long long a, b, word, last_word, sentence_position;
  long long c;
  for (sentence_position = 0; sentence_position < sentence_length; ++sentence_position) {
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
    b = (*next_random) % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += src->syn0[c + last_word * layer1_size];
      }

      ProcessCbow(word, next_random, src, neu1, neu1e);

      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) src->syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;

        ProcessSkipPair(word, last_word, next_random, src, neu1, neu1e);
      } // for a (skipgram)
    } // end if cbow
  } // sentence
}

void *TrainModelThread(void *id) {
  long long word, sentence_length = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  unsigned long long next_random = (long long)id;
  clock_t now;

  real *neu1 = (real *)calloc(layer1_size, sizeof(real)); // cbow
  real *neu1e = (real *)calloc(layer1_size, sizeof(real)); // skipgram
  FILE *fi = fopen(src->train_file, "rb");
  fseek(fi, src->file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      src->word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         src->word_count_actual / (real)(src->train_words + 1) * 100,
         src->word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - src->word_count_actual / (real)(src->train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    sentence_length = 0;
    while (1) {
      word = ReadWordIndex(fi, src->vocab, src->vocab_hash);
      if (feof(fi)) break;
      if (word == -1) continue;
      word_count++;
      if (word == 0) break;
      // The subsampling randomly discards frequent words while keeping the ranking same
      if (sample > 0) {
        real ran = (sqrt(src->vocab[word].cn / (sample * src->train_words)) + 1) * (sample * src->train_words) / src->vocab[word].cn;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
      }
      sen[sentence_length] = word;
      sentence_length++;
      if (sentence_length >= MAX_SENTENCE_LENGTH) break;
    }

    ProcessSentence(sentence_length, sen, src, &next_random, neu1, neu1e);

    if (feof(fi)) break;
    if (word_count > src->train_words / num_threads) break;
  }
  
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

// TODO: if is_separate=1, save output_file.We and output_file.words
void SaveVector(char* output_file, struct train_params *params){
  long a, b;
  long long vocab_size = params->vocab_size;
  struct vocab_word *vocab = params->vocab;
  real *syn0 = params->syn0;
  FILE* fo = fopen(output_file, "wb");
  
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) { // binary
      for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    } else { // text
      for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void KMeans(char* output_file, struct train_params *params){
  long a, b, c, d;
  long long vocab_size = params->vocab_size;
  struct vocab_word *vocab = params->vocab;
  real *syn0 = params->syn0;
  FILE* fo = fopen(output_file, "wb");
  
  // Run K-means on the word vectors
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(vocab_size, sizeof(int));
  real closev, x;
  real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
  for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
      centcn[cl[c]]++;
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
    for (c = 0; c < vocab_size; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  // Save the K-means classes
  for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
  free(centcn);
  free(cent);
  free(cl);
  fclose(fo);
}

void execute(char* command){
  //fprintf(stderr, "# Executing: %s\n", command);
  system(command);
}

void eval_mono(char* emb_file, char* lang, int iter) {
  char command[MAX_STRING];

  /** WordSim **/
  chdir("wordsim/code");
  fprintf(stderr, "\n# eval %d %s %s", iter, lang, "wordSim");
  sprintf(command, "./run_wordSim.sh %s 1 %s", emb_file, lang);
  execute(command);
  chdir("../..");

  /** Analogy **/
  if((iter+1)%10==0 && strcmp(lang, "en")==0){
    chdir("analogy/code");
    fprintf(stderr, "# eval %d %s %s", iter, "en", "analogy");
    sprintf(command, "./run_analogy.sh %s 1", emb_file);
    execute(command);
        chdir("../..");
  }
}

// cross-lingual document classification
void cldc(char* outPrefix, int iter) {
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

void eval_mono_bi(char* iter_prefix, struct train_params *src, struct train_params *tgt, int iter) {
  char command[MAX_STRING];
  char *src_lang = src->lang;
  char *tgt_lang = tgt->lang;
  char src_we_file[MAX_STRING], tgt_we_file[MAX_STRING];

  fprintf(stderr, "\n# Evaluation at iter=%d\n", iter);

  // save embeddings
  sprintf(src_we_file, "%s.%s", output_prefix, src_lang);
  sprintf(tgt_we_file, "%s.%s", output_prefix, tgt_lang);
  SaveVector(src_we_file, src);
  SaveVector(tgt_we_file, tgt);

  // eval mono
  eval_mono(src_we_file, src_lang, iter);
  eval_mono(tgt_we_file, src_lang, iter);

  // eval bi
  if(strcmp(src_lang, "de")==0 && strcmp(tgt_lang, "en")==0){
    // create symlinks with names expected by CLDC
    sprintf(command, "ln -s %s %s.%s-%s.%s", src_we_file, iter_prefix, src_lang, tgt_lang, src_lang);
    sprintf(command, "ln -s %s %s.%s-%s.%s", tgt_we_file, iter_prefix, src_lang, tgt_lang, tgt_lang);

    // cross-lingual document classification
    cldc(iter_prefix, iter);
  }
}

// init for each language
void mono_init(struct train_params *params){
  int i;
  if (params->read_vocab_file[0] != 0) ReadVocab(params); else LearnVocabFromTrainFile(params);
  if (params->save_vocab_file[0] != 0) SaveVocab(params);
  sprintf(params->output_file, "%s.%s", output_prefix, params->lang);
  InitNet(params);
  if (negative > 0) InitUnigramTable(params);
  ComputeBlockStartPoints(params->train_file, num_threads, &params->line_blocks, &params->num_lines);

  // check
  fprintf(stderr, "Done! num_lines=%lld, blocks = [", params->num_lines);
  for(i=0; i<=num_threads; i++) { fprintf(stderr, " %lld", src->line_blocks[i]); }
  fprintf(stderr, " ]\n");
}

void TrainModel() {
  long a;

  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", src->train_file);
  starting_alpha = alpha;
  if (output_prefix[0] == 0) return;

  // init
  mono_init(src);
  if(is_tgt){
    mono_init(tgt);
    assert(src->num_lines==tgt->num_lines);
  }
  if (use_align) {
    ComputeBlockStartPoints(align_file, num_threads, &align_line_blocks, &align_num_lines);
    assert(src->num_lines==align_num_lines);
  }

  start = clock();
  for(cur_iter=start_iter; cur_iter<num_train_iters; cur_iter++){
    fprintf(stderr, "# Start iter %d, num_threads=%d\n", cur_iter, num_threads);

    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

    // src
    if (classes == 0) {
      SaveVector(src->output_file, src);
    } else {
      KMeans(src->output_file, src);
    }
    if (eval_opt>=0) {
      eval_mono(src->output_file, src->lang, cur_iter);
      if (is_tgt) eval_mono(tgt->output_file, tgt->lang, cur_iter);
    }
  }
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

struct train_params *InitTrainParams() {
  struct train_params *params = malloc(sizeof(struct train_params));
  params->save_vocab_file[0] = 0;
  params->read_vocab_file[0] = 0;

  params->train_words = 0;
  params->word_count_actual = 0;
  params->file_size = 0;
  params->num_lines = 0;

  params->vocab_size = 0;
  params->vocab_max_size = 1000;
  params->vocab = (struct vocab_word *)calloc(params->vocab_max_size, sizeof(struct vocab_word));
  params->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  return params;
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

    printf("\t-eval <int>\n");
    printf("\t\t0 -- word sim (default = -1, no evaluation)\n");


    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }

  src = InitTrainParams();
  tgt = InitTrainParams();

  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-src-train", argc, argv)) > 0) strcpy(src->train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-train", argc, argv)) > 0) {
    is_tgt = 1;
    strcpy(tgt->train_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-align", argc, argv)) > 0) {
    use_align = 1;
    strcpy(align_file, argv[i + 1]);
  }

  //if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(src->save_vocab_file, argv[i + 1]);
  //if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(src->read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_prefix, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  // evaluation
  if ((i = ArgPos((char *)"-eval", argc, argv)) > 0) eval_opt = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-src-lang", argc, argv)) > 0) strcpy(src->lang, argv[i + 1]);
  if ((i = ArgPos((char *)"-tgt-lang", argc, argv)) > 0) strcpy(tgt->lang, argv[i + 1]);

  // get absolute path for output_prefix
  char actual_path [MAX_STRING];
  realpath(output_prefix, actual_path);
  strcpy(output_prefix, actual_path);

  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  
  TrainModel();

  // print some debug info
  printf("\n");
  stat(src->syn0, src->vocab_size * layer1_size, (char*) "syn0");
  if (hs) stat(src->syn1, src->vocab_size * layer1_size, (char*) "syn1");
  if (negative) stat(src->syn1neg, src->vocab_size * layer1_size, (char*) "syn1neg");

  return 0;
}
