/*modifica lexer su slide18 presentazione04
-riconosci altri 2 op aritmetici
-parentesi
-identificatori
-parola chiave for*/

%{
#include <iostream>
using namespace std;
%}

DIGIT   [0-9]
LETTER  [A-Za-z]
SEP     [\s\t\n]

/* read only one input file */
%option noyywrap

%%

"+"               { cout << "        operatore <" << yytext[0] << ">" << endl; }
"-"               { cout << "        operatore <" << yytext[0] << ">" << endl; }
"/"               { cout << "        operatore <" << yytext[0] << ">" << endl; }
"*"               { cout << "        operatore <" << yytext[0] << ">" << endl; }
"="               { cout << "        operatore <" << yytext[0] << ">" << endl; }

"["               { cout << "        operatore <" << yytext[0] << ">" << endl; }
"]"               { cout << "        operatore <" << yytext[0] << ">" << endl; }

"for"             { cout << "        keyword <" << yytext << ">" << endl; }

{SEP}*  

{LETTER}({LETTER}|{DIGIT})*                { cout << "        identificatore <" << yytext << ">" << endl; }
           
.                 { cout << "Token sconosciuto <" << yytext[0] << ">" << endl; }
%%

int main(int argc, char** argv) {
    FlexLexer* lexer = new yyFlexLexer;
    lexer->yylex();
    return 0;
}
