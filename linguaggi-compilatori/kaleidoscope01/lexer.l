/*  
    kaleidoscope 01
    riconoscere questo linguaggio
    def g(x y z) x*y+z;
    extern f(x y);
    3*f(3,4)-x;
*/

%{
#include <iostream>
using namespace std;
%}

digit   [0-9]
letter  [A-Za-z]
sep     [\s\t\n]
op      [+-/*]

/* read only one input file */
%option noyywrap

%%

{op}    { cout << "operatore <" << yytext << ">\n"; }

"def"   { cout << "keyword <" << yytext << ">\n"; }
"extern"   { cout << "        keyword <" << yytext << ">" << endl; }

{sep}*  {}              

{LETTER}({LETTER}|{DIGIT})*                { cout << "        identificatore <" << yytext << ">" << endl; }
           
.                 { cout << "Token sconosciuto <" << yytext[0] << ">" << endl; }

%%

int main(int argc, char** argv) {
    FlexLexer* lexer = new yyFlexLexer;
    lexer->yylex();
    return 0;
}
