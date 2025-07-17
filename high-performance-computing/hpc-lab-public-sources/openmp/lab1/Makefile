ifndef EXERCISE
EXERCISE=exercise1.c
endif

CC=clang
LD=ld
OBJDUMP=objdump

OPT=-O2 -g -fopenmp
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

SRCS=main.c utils.c
OBJS=$(SRCS:.c=.o) $(EXERCISE:.c=.o)
EXE=$(EXERCISE:.c=.exe)

$(EXE):	$(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

all: $(EXE)

.PHONY: run clean
run: $(EXE)
	./$(EXE)

clean:
	rm -f $(OBJS) *.o *.exe *.out *~

