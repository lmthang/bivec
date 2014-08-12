#include <stdio.h>

int main(int argc, char *argv[]) {
  char command[100];
  FILE *output;
  int numLines;
  sprintf(command, "wc -l %s", argv[1]);

  output = popen(command, "r");
  fscanf(output, "%d", &numLines);

  printf("%d\n", numLines);
  return 0;
}
