#define EPS 0
#define BASE_VARIABLES 256

const int numexpr = 256;
const int expr = 257;
const int parexpr = 258;
const int idexpr = 259;
const int optexpr = 260;
const int exprlist = 261;
const int primary = 262;
const int binop = 263;
const int binoprhs = 264;
const int idseq = 265;
const int proto = 266;
const int def = 267;
const int external = 268;
const int top = 269;
const int program = 270;
const int AXIOM = 270;

std::vector<int> prod1{-2};	// numexpr : 'number'
std::vector<int> prod2{40,257,41};	// parexpr : '(' expr ')'
std::vector<int> prod3{-3};	// idexpr : 'id'
std::vector<int> prod4{-3,40,260,41};	// idexpr : 'id' '(' optexpr ')'
