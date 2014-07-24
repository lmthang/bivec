#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define MAX_STRING 1000

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

int main(int argc, char **argv) {
  if (argc == 1) {
    printf("runCLDC <prefix>\n\n");
    exit(1);
  }

  char out_prefix[MAX_STRING];
  strcpy(out_prefix, argv[1]);
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/klementiev-40/original";
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/hermann-128/add";
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/hermann-128/bi";
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/hermann-128/bi_plus";
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/hermann-128/add_plus";
//  char *outPrefix = (char*)"/Users/phamhyhieu/Code/word2vec/data/sarath-40/sarath.40";

  printf("# Run CLDC on %s\n", out_prefix);
  cldcEvaluate(out_prefix, 0);

  return 0;
}
