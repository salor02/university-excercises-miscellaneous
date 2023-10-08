build: lexer

lexer.o: lexer.cc
	flex -+ -o lexer.cc lexer.l
	g++ -c lexer.cc

main.o: main.cc
	g++ -c main.cc

lexer: main.o lexer.o
	g++ -o lexer main.o lexer.o

.PHONY: clean

clean:
	rm -rf *.o