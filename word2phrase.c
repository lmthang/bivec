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

#define MAX_STRING 60

const int vocab_hash_size = 500000000; // Maximum 500M entries in the vocabulary

typedef float real;                    // Precision of float numbers

static const char bigram_sep[] = "#";

struct vocab_word {
  long long cn;
  char *word;
  char is_bigram; // Thang: indicate if this is a bigram
};

// training structure, useful when training embeddings for multiple languages
struct train_params {
  char train_file[MAX_STRING];
  char output_file[MAX_STRING];
  char vocab_file[MAX_STRING];
  char config_file[MAX_STRING];
  struct vocab_word *vocab;
  int *vocab_hash;
  long long train_words, word_count_actual, file_size;

  // syn0: input embeddings (both hs and negative)
  // syn1: output embeddings (hs)
  // syn1neg: output embeddings (negative)
  // table, vocab_size corresponds to the output side.
  long long vocab_max_size, vocab_size;
  real *syn0, *syn1, *syn1neg;
  int *table;

  // line blocks
  long long num_lines;
  long long *line_blocks;

  long long unk_id; // index of the <unk> word
};

struct train_params *InitTrainParams() {
  struct train_params *params = malloc(sizeof(struct train_params));

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

struct train_params *params;
struct train_params *short_list_params;

char output_file[MAX_STRING];

// char train_file[MAX_STRING];
// struct vocab_word *vocab;
// long long train_words = 0;
// long long vocab_size = 0;
// int *vocab_hash;

int debug_mode = 2, min_count = 5, min_reduce = 1;
long long vocab_max_size = 10000;
real threshold = 100;

unsigned long long next_random = 1;

// short list
char short_list_file[MAX_STRING];
struct vocab_word *short_list;
int is_short_list = 0;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// Return word length
int ReadWord(char *word, FILE *fin) {
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
        return 4;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;

  return a;
}

// Returns hash value of a word
int GetWordHash(const char *word) {
  unsigned long long a, hash = 1;
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
  int word_len = ReadWord(word, fin);
  if(word_len >= MAX_STRING - 2) printf("! long word: %s\n", word);

  if (feof(fin)) return -1;
  return SearchVocab(word, vocab, vocab_hash);
}

// Adds a word to the vocabulary
int AddWordToVocab(const char *word, struct train_params *params, char is_bigram) {
  unsigned int hash, length = strlen(word) + 1;
  long long vocab_size = params->vocab_size;
  long long vocab_max_size = params->vocab_max_size;
  struct vocab_word *vocab = params->vocab;
  int *vocab_hash = params->vocab_hash;

  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab[vocab_size].is_bigram = is_bigram;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 10000;
    vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash]=vocab_size - 1;
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
void SortVocab(struct train_params *params, int min_count) {

  int a, size;
  unsigned int hash;
  int *vocab_hash = params->vocab_hash;
  struct vocab_word *vocab = params->vocab;
  long long vocab_size = params->vocab_size;

  // Sort the vocabulary and keep </s> at the first position
  printf("  Sorting vocab, size %lld ...", vocab_size); fflush(stdout);
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  params->train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)){ // a=0 is </s> and we want to keep it.
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      params->train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  params->vocab = vocab;
  params->vocab_size = vocab_size;
  printf(" Done\n"); fflush(stdout);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct train_params *params) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < params->vocab_size; a++) if (params->vocab[a].cn > min_reduce) {
    params->vocab[b].cn = params->vocab[a].cn;
    params->vocab[b].word = params->vocab[a].word;
    params->vocab[b].is_bigram = params->vocab[a].is_bigram;
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

void LearnVocabFromTrainFile(struct train_params *params, int min_count) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
 
  // bigram
  char last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  long long start = 1;

  if (debug_mode > 0) printf("# Learn vocab from %s\n", params->train_file);

  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  fin = fopen(params->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  params->vocab_size = 0;
  AddWordToVocab((char *)"</s>", params, 0);
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;

    // bigram
    if (!strcmp(word, "</s>")) {
      start = 1;
      continue;
    } else start = 0;

    params->train_words++;
    if ((debug_mode > 1) && (params->train_words % 100000 == 0)) {
      printf("%lldK%c", params->train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word, params->vocab, params->vocab_hash);

    if (i == -1) {
      a = AddWordToVocab(word, params, 0);
      params->vocab[a].cn = 1;
    } else params->vocab[i].cn++;

    // bigram
    if (start) continue;
    sprintf(bigram_word, "%s%s%s", last_word, bigram_sep, word);
    bigram_word[MAX_STRING - 1] = 0;
    strcpy(last_word, word);
    i = SearchVocab(bigram_word, params->vocab, params->vocab_hash);
    if (i == -1) {
      a = AddWordToVocab(bigram_word, params, 1);
      params->vocab[a].cn = 1;
    } else params->vocab[i].cn++;
 
    if (params->vocab_size > vocab_hash_size * 0.7) ReduceVocab(params);
  }

  SortVocab(params, min_count);
  printf("  Vocab size: %lld\n", params->vocab_size);
  printf("  Words in train file: %lld\n", params->train_words);
  fflush(stdout);  

  params->file_size = ftell(fin);
  fclose(fin);
}

void TrainModel() {
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  real score;
  FILE *fo, *fin;
  printf("Starting training using file %s\n", params->train_file);
  LearnVocabFromTrainFile(params, min_count);

  if (is_short_list) LearnVocabFromTrainFile(short_list_params, 0);

  // Thang: output bigrams
  char bigram_file[MAX_STRING];
  sprintf(bigram_file, "%s.bigram", params->output_file);
  printf("# Print bigrams to %s\n", bigram_file);
  FILE *fo_bigram = fopen(bigram_file, "wb");
  char *word1, *word2;
  int a;
  int bigram_count = 0;
  for (a = 0; a < params->vocab_size; a++) {
    if (params->vocab[a].is_bigram == 0) continue;
    // check if this is really a bigram according to our criteria
    char *bigram = malloc(MAX_STRING);
    strcpy(bigram, params->vocab[a].word);
    word1 = strsep(&bigram, bigram_sep);
    word2 = strsep(&bigram, bigram_sep);
    oov = 0;
    i = SearchVocab(word1, params->vocab, params->vocab_hash);
    if (i == -1) oov = 1; else pa = params->vocab[i].cn;
    i = SearchVocab(word2, params->vocab, params->vocab_hash);
    if (i == -1) oov = 1; else pb = params->vocab[i].cn;
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    pab = params->vocab[a].cn;
    if (oov) score = 0; else score = (pab - min_count) / (real)pa / (real)pb * (real)params->train_words;
    if (score > threshold) {
      if (is_short_list && SearchVocab(word1, short_list_params->vocab, short_list_params->vocab_hash) == -1
                        && SearchVocab(word2, short_list_params->vocab, short_list_params->vocab_hash) == -1) {
        continue;
      }
      fprintf(fo_bigram, "%s\n", params->vocab[a].word);
      bigram_count++;
    }
  }
  printf("  Done, num of bigrams %d\n", bigram_count);
  fclose(fo_bigram);

  fin = fopen(params->train_file, "rb");
  printf("# Output to %s\n", params->output_file);
  fo = fopen(params->output_file, "wb");
  word[0] = 0;
  while (1) {
    strcpy(last_word, word);
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      fprintf(fo, "\n");
      continue;
    }
    cn++;
    if ((debug_mode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    oov = 0;
    i = SearchVocab(word, params->vocab, params->vocab_hash);
    if (i == -1) oov = 1; else pb = params->vocab[i].cn;
    if (li == -1) oov = 1;
    li = i;
    sprintf(bigram_word, "%s%s%s", last_word, bigram_sep, word);
    bigram_word[MAX_STRING - 1] = 0;
    i = SearchVocab(bigram_word, params->vocab, params->vocab_hash);
    if (i == -1) oov = 1; else pab = params->vocab[i].cn;
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    if (oov) score = 0; else score = (pab - min_count) / (real)pa / (real)pb * (real)params->train_words;
    if (score > threshold) {
      fprintf(fo, "%s%s", bigram_sep, word);
      pb = 0;
    } else fprintf(fo, " %s", word);
    pa = pb;
  }
  fclose(fo);
  fclose(fin);
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
  int i;
  if (argc == 1) {
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-short-list <file>\n");
    printf("\t\tSelect phrases that contain words in the short list\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }

  params = InitTrainParams();
  short_list_params = InitTrainParams();

  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(params->train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-short-list", argc, argv)) > 0) {
    strcpy(short_list_params->train_file, argv[i + 1]);
    is_short_list = 1;
  }
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(params->output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  TrainModel();
  return 0;
}
