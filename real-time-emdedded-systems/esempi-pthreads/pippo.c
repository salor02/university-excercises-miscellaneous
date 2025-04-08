extern int i;

int i;

int f(int x)
{
  int *xxx;
  int locale;
  locale = i+x;

  xxx = (int *)0;
  *xxx=54;

  return locale;
}

int main(int argc, char **argv)
{
 return f(10);

}
