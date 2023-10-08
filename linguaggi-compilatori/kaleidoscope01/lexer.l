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
sep     [ \t\n]
op      [+-/*=]
id      {letter}({letter}|{digit})*

/* read only one input file */
%option noyywrap

%%

{op}        { cout << "operand <" << yytext << ">\n"; }

"("         { cout << "open bracket <" << yytext << ">\n"; }
")"         { cout << "closed bracket <" << yytext << ">\n"; }
";"         { cout << "semicolon <" << yytext << ">\n"; }

"def"       { cout << "keyword <" << yytext << ">\n"; }
"extern"    { cout << "keyword <" << yytext << ">\n"; }

{digit}*    { cout << "number <" << yytext << ">\n"; }

{sep}*      {}              

{id}        { cout << "identificatore <" << yytext << ">\n"; }
           
.           { cout << "Token sconosciuto <" << yytext << ">\n"; }

%%
