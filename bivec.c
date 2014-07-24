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

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
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
  char train_file[MAX_STRING], output_file[MAX_STRING];
  char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
  struct vocab_word *vocab;
  int *vocab_hash;
  long long train_words, word_count_actual, file_size;
  long long vocab_max_size, vocab_size;
  real *syn0, *syn1, *syn1neg;
  
  int *table;
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
struct train_params *src, *tgt;

void InitUnigramTable(struct train_params *params) {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  params->table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < params->vocab_size; a++) train_words_pow += pow(params->vocab[a].cn, power);
  i = 0;
  d1 = pow(params->vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    params->table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(params->vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= params->vocab_size) i = params->vocab_size - 1;
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
int SearchVocab(char *word, struct train_params *params) {
  unsigned int hash = GetWordHash(word);
  if (strcmp(word, "</s>") == 0) printf("%u\n", hash);
  while (1) {
    printf("%d\n", params->vocab_hash[hash]);
    printf("%s\n", params->vocab[params->vocab_hash[hash]].word);
    if (params->vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, params->vocab[params->vocab_hash[hash]].word)) return params->vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, struct train_params *params) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word, params);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, struct train_params *params) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  params->vocab[params->vocab_size].word = (char *)calloc(length, sizeof(char));
  printf("Add %s, position %d\n", word, params->vocab_size);
  strcpy(params->vocab[params->vocab_size].word, word);
  printf("Add %s\n", params->vocab[params->vocab_size].word);
  params->vocab[params->vocab_size].cn = 0;
  params->vocab_size++;
  // Reallocate memory if needed
  if (params->vocab_size + 2 >= params->vocab_max_size) {
    params->vocab_max_size += 1000;
    params->vocab = (struct vocab_word *)realloc(params->vocab, params->vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  printf("hash %d\n", hash);
  while (params->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  params->vocab_hash[hash] = params->vocab_size - 1;
  printf("map %d -> %d\n", hash);
  return params->vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct train_params *params) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&params->vocab[1], params->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  size = params->vocab_size;
  params->train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (params->vocab[a].cn < min_count) {
      params->vocab_size--;
      free(params->vocab[params->vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(params->vocab[a].word);
      while (params->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      params->vocab_hash[hash] = a;
      params->train_words += params->vocab[a].cn;
    }
  }
  params->vocab = (struct vocab_word *)realloc(params->vocab, (params->vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < params->vocab_size; a++) {
    params->vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    params->vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
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
    printf("%s\n", word);
    if (feof(fin)) break;
    params->train_words++;
    if ((debug_mode > 1) && (params->train_words % 100000 == 0)) {
      printf("%lldK%c", params->train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word, params);
    printf("HERE\n");
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
    a = posix_memalign((void **)&params->syn1, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
     params->syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&params->syn1neg, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
     params->syn1neg[a * layer1_size + b] = 0;
  }
  for (b = 0; b < layer1_size; b++) for (a = 0; a < params->vocab_size; a++)
   params->syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  CreateBinaryTree(params);
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
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
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, src);
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
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > src->train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
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
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = src->table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (src->vocab_size - 1) + 1;
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
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = src->table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (src->vocab_size - 1) + 1;
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
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void SaveVector(struct train_params *params){
  long a, b;
  FILE* fo = fopen(params->output_file, "wb");
  
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", params->vocab_size, layer1_size);
  for (a = 0; a < params->vocab_size; a++) {
    fprintf(fo, "%s ", params->vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&params->syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", params->syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void KMeans(struct train_params *params){
  long a, b, c, d;
  FILE* fo = fopen(params->output_file, "wb");
  
  // Run K-means on the word vectors
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(params->vocab_size, sizeof(int));
  real closev, x;
  real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
  for (a = 0; a < params->vocab_size; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < params->vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += params->syn0[c * layer1_size + d];
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
    for (c = 0; c < params->vocab_size; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * params->syn0[c * layer1_size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  // Save the K-means classes
  for (a = 0; a < params->vocab_size; a++) fprintf(fo, "%s %d\n", params->vocab[a].word, cl[a]);
  free(centcn);
  free(cent);
  free(cl);
  fclose(fo);
}

void TrainModel() {
  long a;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", src->train_file);
  starting_alpha = alpha;
  if (src->read_vocab_file[0] != 0) ReadVocab(src); else LearnVocabFromTrainFile(src);
  printf("Read dictionary.\n");
  if (src->save_vocab_file[0] != 0) SaveVocab(src);
  printf("Save dictionary.\n");
  if (src->output_file[0] == 0) return;
  InitNet(src);
  if (negative > 0) InitUnigramTable(src);
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  if (classes == 0) {
    SaveVector(src);
  } else {
    KMeans(src);
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
  params->train_words = 0;
  params->word_count_actual = 0;
  params->file_size = 0;
  params->vocab_max_size = 0;
  params->vocab_size = 0;
  params->output_file[0] = 0;
  params->save_vocab_file[0] = 0;
  params->read_vocab_file[0] = 0;
  return params;
}

int main(int argc, char **argv) {
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
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }

  src = InitTrainParams();

  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(src->train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(src->save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(src->read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(src->output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  src->vocab = (struct vocab_word *)calloc(src->vocab_max_size, sizeof(struct vocab_word));
  src->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
