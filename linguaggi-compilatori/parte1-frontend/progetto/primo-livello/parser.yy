%skeleton "lalr1.cc" /* -*- C++ -*- */
%require "3.2"
%defines

%define api.token.constructor
%define api.location.file none
%define api.value.type variant
%define parse.assert

%code requires {
  # include <string>
  #include <exception>
  class driver;
  class RootAST;
  class ExprAST;
  class NumberExprAST;
  class VariableExprAST;
  class CallExprAST;
  class FunctionAST;
  class SeqAST;
  class PrototypeAST;
  class GlobalVarAST;
  class BlockExprAST;
  class VarBindingAST;
  class StmtAST;
  class AssignmentAST;
}

// The parsing context.
%param { driver& drv }

%locations

%define parse.trace
%define parse.error verbose

%code {
# include "driver.hpp"
}

%define api.token.prefix {TOK_}
%token
  END  0  "end of file"
  SEMICOLON  ";"
  COMMA      ","
  MINUS      "-"
  PLUS       "+"
  STAR       "*"
  SLASH      "/"
  LPAREN     "("
  RPAREN     ")"
  QMARK	     "?"
  COLON      ":"
  LT         "<"
  EQ         "=="
  ASSIGN     "="
  LBRACE     "{"
  RBRACE     "}"
  EXTERN     "extern"
  DEF        "def"
  VAR        "var"
  GLOBAL     "global"
;
%token <std::string> IDENTIFIER "id"
%token <double> NUMBER "number"

%type <ExprAST*> exp
%type <ExprAST*> idexp
%type <ExprAST*> expif
%type <ExprAST*> condexp
%type <ExprAST*> initexp
%type <std::vector<ExprAST*>> optexp
%type <std::vector<ExprAST*>> explist
%type <RootAST*> program
%type <RootAST*> top
%type <std::vector<StmtAST*>> stmts
%type <StmtAST*> stmt
%type <RootAST*> assignment
%type <FunctionAST*> definition
%type <PrototypeAST*> external
%type <GlobalVarAST*> globalvar
%type <PrototypeAST*> proto
%type <std::vector<std::string>> idseq
%type <BlockExprAST*> block
%type <std::vector<VarBindingAST*>> vardefs
%type <VarBindingAST*> binding


%%
%start startsymb;

startsymb:
    program                 { drv.root = $1; }

program:
    %empty               { $$ = new SeqAST(nullptr,nullptr); }
|   top ";" program      { $$ = new SeqAST($1,$3); };

//added in first-level: 
//  globalvar
top:
    %empty                { $$ = nullptr; }
|   definition            { $$ = $1; }
|   external              { $$ = $1; }
|   globalvar             { $$ = $1; };  

//modified in first-level: 
//  "def" proto exp -> "def" proto block
definition:
    "def" proto block       { $$ = new FunctionAST($2,$3); $2->noemit(); };

external:
    "extern" proto        { $$ = $2; };

proto:
    "id" "(" idseq ")"    { $$ = new PrototypeAST($1,$3);  };

//added in first-level
globalvar:
    "global" "id"         { $$ = new GlobalVarAST($2); }      

idseq:
    %empty                { std::vector<std::string> args;
                         $$ = args; }
|   "id" idseq            { $2.insert($2.begin(),$1); $$ = $2; };

%left ":";
%left "<" "==";
%left "+" "-";
%left "*" "/";

//added in first-level
stmts:
    stmt                   { std::vector<StmtAST*> stmts;
                            stmts.insert(stmts.begin(), $1);
                            $$ = stmts; }
|   stmt ";" stmts        { $3.insert($3.begin(), $1);
                            $$ = $3; }

//added in first-level
stmt:
    assignment            { $$ = new StmtAST($1); }
|   block                 { $$ = new StmtAST($1); }
|   exp                   { $$ = new StmtAST($1); };

//added in first-level
assignment:
    "id" "=" exp          { $$ = new AssignmentAST($1,$3); };

//modified in first-level:
//  blockexp -> block
//  "{" vardefs ";" exp "}" -> "{" vardefs ";" stmts "}"
//added in first-level: 
//  "{" stmts "}" 
block:
    "{" stmts "}"             { std::vector<VarBindingAST*> dummy; 
                                $$ = new BlockExprAST(dummy, $2);}
|   "{" vardefs ";" stmts "}" { $$ = new BlockExprAST($2,$4); }

vardefs:
    binding                 { std::vector<VarBindingAST*> definitions;
                            definitions.push_back($1);
                            $$ = definitions; }
|   vardefs ";" binding     { $1.push_back($3);
                            $$ = $1; }

//modified in first-level:
//  "var" "id" "=" exp -> "var" "id" initexp
binding:
    "var" "id" initexp     { $$ = new VarBindingAST($2,$3); }

//modified in first-level:
//  removed blockexp    
exp:
    exp "+" exp           { $$ = new BinaryExprAST('+',$1,$3); }
|   exp "-" exp           { $$ = new BinaryExprAST('-',$1,$3); }
|   exp "*" exp           { $$ = new BinaryExprAST('*',$1,$3); }
|   exp "/" exp           { $$ = new BinaryExprAST('/',$1,$3); }
|   idexp                 { $$ = $1; }
|   "(" exp ")"           { $$ = $2; }
|   "number"              { $$ = new NumberExprAST($1); }
|   expif                 { $$ = $1; };

//added in first-level
initexp:
    %empty                { $$ = nullptr; }
|   "=" exp               { $$ = $2; };
                      
expif:
    condexp "?" exp ":" exp { $$ = new IfExprAST($1,$3,$5); }

condexp:
    exp "<" exp           { $$ = new BinaryExprAST('<',$1,$3); }
|   exp "==" exp          { $$ = new BinaryExprAST('=',$1,$3); }

idexp:
    "id"                  { $$ = new VariableExprAST($1); }
|   "id" "(" optexp ")"   { $$ = new CallExprAST($1,$3); };

optexp:
    %empty                { std::vector<ExprAST*> args;
			                $$ = args; }
|   explist               { $$ = $1; };

explist:
    exp                   { std::vector<ExprAST*> args;
                            args.push_back($1);
			                $$ = args;
                            }
|   exp "," explist       { $3.insert($3.begin(), $1); $$ = $3; };
 
%%

void
yy::parser::error (const location_type& l, const std::string& m)
{
  std::cerr << l << ": " << m << '\n';
}
