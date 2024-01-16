#ifndef DRIVER_HPP
#define DRIVER_HPP
/************************* IR related modules ******************************/
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
/**************** C++ modules and generic data types ***********************/
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <variant>

#include "parser.hpp"

using namespace llvm;

// Dichiarazione del prototipo yylex per Flex
// Flex va proprio a cercare YY_DECL perché
// deve espanderla (usando M4) nel punto appropriato
# define YY_DECL \
  yy::parser::symbol_type yylex (driver& drv)
// Per il parser è sufficiente una forward declaration
YY_DECL;

// Classe che organizza e gestisce il processo di compilazione
class driver{
public:
    driver();
    std::map<std::string, AllocaInst*> NamedValues; // Tabella associativa in cui ogni 
                // chiave x è una variabile e il cui corrispondente valore è un'istruzione 
                // che alloca uno spazio di memoria della dimensione necessaria per 
                // memorizzare un variabile del tipo di x (nel nostro caso solo double)
    RootAST* root;      // A fine parsing "punta" alla radice dell'AST
    int parse (const std::string& f);
    std::string file;
    bool trace_parsing; // Abilita le tracce di debug el parser
    void scan_begin (); // Implementata nello scanner
    void scan_end ();   // Implementata nello scanner
    bool trace_scanning;// Abilita le tracce di debug nello scanner
    yy::location location; // Utillizata dallo scannar per localizzare i token
    void codegen();
};

typedef std::variant<std::string,double> lexval;
const lexval NONE = 0.0;

// Classe base dell'intera gerarchia di classi che rappresentano
// gli elementi del programma
class RootAST {
public:
    virtual ~RootAST() {};
    virtual lexval getLexVal() const {return NONE;};
    virtual Value *codegen(driver& drv) { return nullptr; };
};

/// SeqAST - Classe che rappresenta la sequenza di statement
class SeqAST : public RootAST {
private:
    RootAST* first;
    RootAST* continuation;

public:
    SeqAST(RootAST* first, RootAST* continuation);
    Value *codegen(driver& drv) override;
};

/// StmtAST - Classe per la rappresentazione degli statement all'interno dei blocchi
class StmtAST : public RootAST {
private:
    RootAST* stmt;

public:
    StmtAST(RootAST* stmt);
    Value *codegen(driver& drv) override;
};

/// GlobalVarAST - Classe per la rappresentazione delle variabili globali
class GlobalVarAST : public RootAST {
protected:
    std::string Name;

public:
    GlobalVarAST(const std::string& Name);
    lexval getLexVal() const override;
    virtual Value *codegen(driver &drv) override;
};

/*  GlobalArrayAST - Classe per la rappresentazione degli array globali, estende la classe
    GlobalVarAST dato che l'unica cosa che aggiunge è il numero di item oltre ad
    avere una codegen leggermente diversa */
class GlobalArrayAST : public GlobalVarAST {
private:
    double itemNum;

public:
    GlobalArrayAST(const std::string& Name, double itemNum);
    Value *codegen(driver &drv) override;
};

/// ExprAST - Classe base per tutti i nodi espressione
class ExprAST : public RootAST {};

/// NumberExprAST - Classe per la rappresentazione di costanti numeriche
class NumberExprAST : public ExprAST {
private:
    double Val;

public:
    NumberExprAST(double Val);
    lexval getLexVal() const override;
    Value *codegen(driver& drv) override;
};

/// VariableExprAST - Classe per la rappresentazione di riferimenti a variabili
class VariableExprAST : public ExprAST {
private:
    std::string Name;
  
public:
    VariableExprAST(const std::string &Name);
    lexval getLexVal() const override;
    Value *codegen(driver& drv) override;
};

/// ArrayExprAST - Classe per la rappresentazione di riferimenti ad array
class ArrayExprAST : public ExprAST {
private:
    std::string Name;
    ExprAST* Idx;
  
public:
    ArrayExprAST(const std::string &Name, ExprAST* Idx);
    lexval getLexVal() const override;
    Value *codegen(driver& drv) override;
};

/// BinaryExprAST - Classe per la rappresentazione di operatori binari
class BinaryExprAST : public ExprAST {
private:
    char Op;
    ExprAST* LHS;
    ExprAST* RHS;

public:
    BinaryExprAST(char Op, ExprAST* LHS, ExprAST* RHS);
    Value *codegen(driver& drv) override;
};

/// UnaryExprAST - Classe per la rappresentazione di operatori unari
class UnaryExprAST : public ExprAST {
private:
    char Op;
    ExprAST* Val;

public:
    UnaryExprAST(char Op, ExprAST* Val);
    Value *codegen(driver& drv) override;
};

/// CallExprAST - Classe per la rappresentazione di chiamate di funzione
class CallExprAST : public ExprAST {
private:
    std::string Callee;
    std::vector<ExprAST*> Args;

public:
    CallExprAST(std::string Callee, std::vector<ExprAST*> Args);
    lexval getLexVal() const override;
    Value *codegen(driver& drv) override;
};

/// IfExprAST - Classe per la rappresentazione dell'expression if
class IfExprAST : public ExprAST {
private:
    ExprAST* Cond;
    ExprAST* TrueExp;
    ExprAST* FalseExp;
public:
    IfExprAST(ExprAST* Cond, ExprAST* TrueExp, ExprAST* FalseExp);
    Value *codegen(driver& drv) override;
};

/// IfExprAST - Classe per la rappresentazione dell'if "normale" (con statement)
class IfStmtAST : public RootAST {
private:
    ExprAST* Cond;
    StmtAST* TrueStmt;
    StmtAST* FalseStmt;
public:
    IfStmtAST(ExprAST* Cond, StmtAST* TrueStmt, StmtAST* FalseStmt);
    Value *codegen(driver& drv) override;
};

/// ForStmtAST - Classe per la rappresentazione di un ciclo for
class ForStmtAST : public RootAST{
private:
    InitAST* Init;
    ExprAST* Cond;
    RootAST* Assignment;
    StmtAST* Stmt;
public:
    ForStmtAST(InitAST* Init, ExprAST* Cond, RootAST* Assignment, StmtAST* Stmt);
    Value *codegen(driver& drv) override;
};

/// BlockAST - Classe per la rappresentazione dei blocchi (istruzioni delimitate da {})
class BlockAST : public ExprAST {
private:
    std::vector<BindingAST*> Def;
    std::vector<StmtAST*> Stmts;
public:
    BlockAST(std::vector<BindingAST*> Def, std::vector<StmtAST*> Stmts);
    Value *codegen(driver& drv) override;
}; 

/// InitAST - Classe per gestire le variabili
/*  Questa è una classe virtuali con funzioni pure virtual (da sovrascrivere necessariamente in classi figlie).
    Grazie a questa classe è possibile gestire la fase di inizializzazione del for, ovvero distinguere il caso in cui 
    avvenga una definizione, con conseguente necessità di allocare nuova memoria, e il caso in cui avvenga un assegnamento
    in cui occorre soltanto creare una store in una locazione di memoria preesistente. Un oggetto di tipo InitAST tratterà 
    sicuramente una variabile ma genererà codice diverso in base al flag definition, impostato a true quando un oggetto della
    classe BindingAST viene instanziato */
class InitAST: public RootAST {  
protected:
    const std::string Name;
    bool definition; //definition viene scritto da oggetti di VarBindingAST e AssignmentAST al momento della creazione
public:
    InitAST(const std::string Name, bool definition);
    bool isDefinition();
    const std::string& getName() const;
    virtual Value *codegen(driver& drv) override = 0;
}; 

/// BindingAST - Classe per la definizione di variabili e array
class BindingAST: public InitAST {
public:
    BindingAST(const std::string Name);
public:
    virtual AllocaInst *codegen(driver& drv) override = 0;
};

/// VarBindingAST - Classe per la definizione di variabili
class VarBindingAST: public BindingAST {
private:
    ExprAST* Val;
public:
    VarBindingAST(const std::string Name, ExprAST* Val);
    AllocaInst *codegen(driver& drv) override;
};

/// ArrayBindingAST - Classe per la definizione di array
class ArrayBindingAST: public BindingAST {
private:
    double itemNum;
    std::vector<ExprAST*> initExprList;
public:
    ArrayBindingAST(const std::string Name, double itemNum, std::vector<ExprAST*> initExprList = std::vector<ExprAST*>());
    AllocaInst *codegen(driver& drv) override;
};

/// AssignmentAST - Classe per l'assegnamento di variabili e array
class AssignmentAST: public InitAST {
public:
    AssignmentAST(const std::string Name);
};

/// VarAssignmentAST - Classe per la gestione degli assegnamenti di variabili
class VarAssignmentAST: public AssignmentAST {
private:
    ExprAST* Val;
public:
    VarAssignmentAST(const std::string Name, ExprAST* Val);
    Value *codegen(driver& drv) override;
};

/// ArrayAssignmentAST - Classe per la gestione degli assegnamenti
class ArrayAssignmentAST: public AssignmentAST {
private:
    ExprAST* Idx;
    ExprAST* Val;
public:
    ArrayAssignmentAST(const std::string Name, ExprAST* Val, ExprAST* Idx);
    Value *codegen(driver& drv) override;
};

/// PrototypeAST - Classe per la rappresentazione dei prototipi di funzione
/// (nome, numero e nome dei parametri; in questo caso il tipo è implicito
/// perché unico)
class PrototypeAST : public RootAST {
private:
    std::string Name;
    std::vector<std::string> Args;
    bool emitcode;

public:
    PrototypeAST(std::string Name, std::vector<std::string> Args);
    const std::vector<std::string> &getArgs() const;
    lexval getLexVal() const override;
    Function *codegen(driver& drv) override;
    void noemit();
};

/// FunctionAST - Classe che rappresenta la definizione di una funzione
class FunctionAST : public RootAST {
private:
    PrototypeAST* Proto;
    BlockAST* Body;
    bool external;
  
public:
    FunctionAST(PrototypeAST* Proto, BlockAST* Body);
    Function *codegen(driver& drv) override;
};

#endif // ! DRIVER_HH
