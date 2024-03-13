#include "driver.hpp"
#include "parser.hpp"

/*NOTE:
1)  Viene data priorità alle variabili locali (variabili locali oscurano variabili globali) in ogni situazione:
    assegnamento, definizione e valutazione di espressioni
2)  Una variabili definita in una funzione oscura eventuali altre variabili locali con lo stesso nome e, di conseguenza
    oscura anche variabili globali
*/

/*TODO:
- Funzione per controllo variabile esistente
- Commenti valori ritorno
- Albero classi
*/

/*BUG NOTI:
1)  Problema con l'utilizzo del segno - per sottrazioni tra una costante negativa e un'altra costante, è possibile
    esclusivamente sottrarre dall'opposto di una variabile un'altra variabile o una costante
*/

// Generazione di un'istanza per ciascuna della classi LLVMContext,
// Module e IRBuilder. Nel caso di singolo modulo è sufficiente
LLVMContext *context = new LLVMContext;
Module *module = new Module("Kaleidoscope", *context);
IRBuilder<> *builder = new IRBuilder(*context);

Value *LogErrorV(const std::string Str) {
    std::cerr << Str << std::endl;
    return nullptr;
}

/* Il codice seguente sulle prime non è semplice da comprendere.
   Esso definisce una utility (funzione C++) con due parametri:
   1) la rappresentazione di una funzione llvm IR, e
   2) il nome per un registro SSA
   La chiamata di questa utility restituisce un'istruzione IR che alloca un double
   in memoria e ne memorizza il puntatore in un registro SSA cui viene attribuito
   il nome passato come secondo parametro. L'istruzione verrà scritta all'inizio
   dell'entry block della funzione passata come primo parametro.
   Si ricordi che le istruzioni sono generate da un builder. Per non
   interferire con il builder globale, la generazione viene dunque effettuata
   con un builder temporaneo TmpB

   NB: modificato nel quarto livello per permettere di passare alla funzione
   i tipi di dato array (se non viene specificato nessun tipo la funzione si comporta 
   normalmente allocando una semplice variabile)
*/
static AllocaInst *CreateEntryBlockAlloca(Function *fun, StringRef VarName, Type* T = Type::getDoubleTy(*context)) {
    IRBuilder<> TmpB(&fun->getEntryBlock(), fun->getEntryBlock().begin());
    return TmpB.CreateAlloca(T, nullptr, VarName);
}

// Implementazione del costruttore della classe driver
driver::driver(): trace_parsing(false), trace_scanning(false) {};

// Implementazione del metodo parse
int driver::parse (const std::string &f) {
    file = f;                    // File con il programma
    location.initialize(&file);  // Inizializzazione dell'oggetto location
    scan_begin();                // Inizio scanning (ovvero apertura del file programma)
    yy::parser parser(*this);    // Istanziazione del parser
    parser.set_debug_level(trace_parsing); // Livello di debug del parsed
    int res = parser.parse();    // Chiamata dell'entry point del parser
    scan_end();                  // Fine scanning (ovvero chiusura del file programma)
    return res;
}

// Implementazione del metodo codegen, che è una "semplice" chiamata del 
// metodo omonimo presente nel nodo root (il puntatore root è stato scritto dal parser)
void driver::codegen() {
    root->codegen(*this);
};

/************************* Sequence tree **************************/
SeqAST::SeqAST(RootAST* first, RootAST* continuation):
    first(first), continuation(continuation) {};

// La generazione del codice per una sequenza è banale:
// mediante chiamate ricorsive viene generato il codice di first e 
// poi quello di continuation (con gli opportuni controlli di "esistenza")
Value *SeqAST::codegen(driver& drv) {
    if (first != nullptr) {
        Value *f = first->codegen(drv);
    }
    else {
        if (continuation == nullptr) return nullptr;
    }
    Value *c = continuation->codegen(drv);
    return nullptr;
};

/********************** Block Sequence tree ***********************/
/*  Per rappresentare gli statements utilizzo una classe apposta perchè c'è bisogno di tornare un puntatore
    a Value dalla funzione chiamante, cosa che non potrei fare usando SeqAST dato che ritorna null in ogni caso */
StmtAST::StmtAST(RootAST* stmt): stmt(stmt) {};

//  Genera solamente il codice dell'istruzione da cui è "composto" lo statement
Value *StmtAST::codegen(driver& drv) {
    return stmt->codegen(drv);
};

/********************* Number Expression Tree *********************/
NumberExprAST::NumberExprAST(double Val): Val(Val) {};

lexval NumberExprAST::getLexVal() const {
  // Non utilizzata, Inserita per continuità con versione precedente
  lexval lval = Val;
  return lval;
};

// Non viene generata un'struzione; soltanto una costante LLVM IR
// corrispondente al valore float memorizzato nel nodo
// La costante verrà utilizzata in altra parte del processo di generazione
// Si noti che l'uso del contesto garantisce l'unicità della costanti 
Value *NumberExprAST::codegen(driver& drv) {  
    return ConstantFP::get(*context, APFloat(Val));
};

/******************** Variable Expression Tree ********************/
VariableExprAST::VariableExprAST(const std::string &Name): Name(Name) {};

lexval VariableExprAST::getLexVal() const {
    lexval lval = Name;
    return lval;
};

// NamedValues è una tabella che ad ogni variabile (che, in Kaleidoscope1.0, 
// può essere solo un parametro di funzione) associa non un valore bensì
// la rappresentazione di una funzione che alloca memoria e restituisce in un
// registro SSA il puntatore alla memoria allocata. Generare il codice corrispondente
// ad una varibile equivale dunque a recuperare il tipo della variabile 
// allocata e il nome del registro e generare una corrispondente istruzione di load
// Negli argomenti della CreateLoad ritroviamo quindi: (1) il tipo allocato, (2) il registro
// SSA in cui è stato messo il puntatore alla memoria allocata (si ricordi che A è
// l'istruzione ma è anche il registro, vista la corrispodenza 1-1 fra le due nozioni), (3)
// il nome del registro in cui verrà trasferito il valore dalla memoria
Value *VariableExprAST::codegen(driver& drv) {
    //priorità ai parametri di funzione, se un nome non è definito nei parametri allora si passa 
    //a controllare le variabili globali
    AllocaInst *A = drv.NamedValues[Name];
    Type *varType;

    /*  In entrambi i casi prima di fare un'eventuale load viene controllato se la variabile indicata 
        è un array */
    if (A){
        varType = A->getAllocatedType();
        if(varType->isArrayTy())
            return LogErrorV("Array utilizzato senza operatore []");
        return builder->CreateLoad(varType, A, Name.c_str());
    }
    else{
        GlobalVariable* globalVar = module->getNamedGlobal(Name);
        if(globalVar){
            varType = globalVar->getValueType();
            if(varType->isArrayTy())
                return LogErrorV("Array utilizzato senza operatore []");
            return builder->CreateLoad(varType, globalVar, Name.c_str());
        }
    }
    return LogErrorV("Variabile "+Name+" non definita");
}

/******************** Variable Expression Tree ********************/
ArrayExprAST::ArrayExprAST(const std::string &Name, ExprAST* Idx): 
    Name(Name), Idx(Idx) {};

lexval ArrayExprAST::getLexVal() const {
    lexval lval = Name;
    return lval;
};

Value *ArrayExprAST::codegen(driver& drv) {

    /*  Anche in questo caso viene data priorità alle variabili locali e in entrambi i casi prima di fare 
        un'eventuale load viene controllato se la variabile indicata è effettivamente un array */

    Type *varType;
    Value *array = drv.NamedValues[Name];
    if (array){
        varType = drv.NamedValues[Name]->getAllocatedType();
        if(!varType->isArrayTy())
            return LogErrorV("La variabile locale " + Name + " non è un array");
    }
    else{
        GlobalVariable* globalVar = module->getNamedGlobal(Name);
        if(globalVar){
            varType = globalVar->getValueType();
            if(!varType->isArrayTy())
                return LogErrorV("La variabile globale " + Name + " non è un array");
            array = globalVar;
        }
        else{
            return LogErrorV("Variabile "+Name+" non definita");
        }
    }

    //generazione codice che definisce il valore dell'indice
    Value *indexVal = Idx->codegen(drv);
    if (!indexVal)  // Qualcosa è andato storto nella generazione del codice?
        return nullptr;

    //il valore prodotto dalla codegen dell'indice deve essere convertito da double ad intero
    indexVal = builder->CreateFPToSI(indexVal, Type::getInt32Ty(*context), "intindex");

    //accesso effettivo alla cella dell'array
    ArrayType *AT = dyn_cast<ArrayType>(varType);
    Value *cell = builder->CreateInBoundsGEP(AT, array, {ConstantInt::get(Type::getInt32Ty(*context), 0), indexVal});

    //creazione della load del valore contenuto nella cella dell'array (il tipo utilizzato è double dato che è l'unico disponibile)
    return builder->CreateLoad(Type::getDoubleTy(*context), cell, Name.c_str());

}

/**************** Global Variable Expression Tree *****************/
GlobalVarAST::GlobalVarAST(const std::string &Name): Name(Name) {};

lexval GlobalVarAST::getLexVal() const {
    lexval lval = Name;
    return lval;
};

Value *GlobalVarAST::codegen(driver& drv) {
    //la variabile globale viene iniziliazzata a zero di default
    Value *globalVar = new GlobalVariable(*module, Type::getDoubleTy(*context), false, GlobalValue::CommonLinkage,  ConstantFP::get(*context, APFloat(0.0)), Name);
    globalVar->print(errs());
    fprintf(stderr, "\n");
    return globalVar;
}

/**************** Global Array Expression Tree *****************/
GlobalArrayAST::GlobalArrayAST(const std::string &Name, double itemNum): 
    GlobalVarAST(Name), itemNum(itemNum) {};

Value *GlobalArrayAST::codegen(driver& drv) {
    //la variabile globale viene iniziliazzata a zero di default
    ArrayType *varType = ArrayType::get(Type::getDoubleTy(*context),itemNum);
    Value *globalVar = new GlobalVariable(*module, varType, false, GlobalValue::CommonLinkage, ConstantAggregateZero::get(varType), Name);
    globalVar->print(errs());
    fprintf(stderr, "\n");
    return globalVar;
}

/******************** Binary Expression Tree **********************/
BinaryExprAST::BinaryExprAST(char Op, ExprAST* LHS, ExprAST* RHS):
    Op(Op), LHS(LHS), RHS(RHS) {};

// La generazione del codice in questo caso è di facile comprensione.
// Vengono ricorsivamente generati il codice per il primo e quello per il secondo
// operando. Con i valori memorizzati in altrettanti registri SSA si
// costruisce l'istruzione utilizzando l'opportuno operatore
Value *BinaryExprAST::codegen(driver& drv) {
    Value *L = LHS->codegen(drv);
    Value *R = RHS->codegen(drv);
    if (!L || !R) 
        return nullptr;
    switch (Op) {
    case '+':
        return builder->CreateFAdd(L,R,"addres");
    case '-':
        return builder->CreateFSub(L,R,"subres");
    case '*':
        return builder->CreateFMul(L,R,"mulres");
    case '/':
        return builder->CreateFDiv(L,R,"addres");
    case '<':
        return builder->CreateFCmpULT(L,R,"lttest");
    case '=':
        return builder->CreateFCmpUEQ(L,R,"eqtest");
    case '&':
        return builder->CreateAnd(L,R,"andres");
    case '|':
        return builder->CreateOr(L,R,"orres");
    default:  
        std::cout << Op << std::endl;
        return LogErrorV("Operatore binario non supportato");
    }
};

/******************** Unary Expression Tree **********************/
UnaryExprAST::UnaryExprAST(char Op, ExprAST* Val):
    Op(Op), Val(Val) {};

/*  Funzionamento del tutto analogo a espressioni binarie con l'unica differenza 
    che il nodo ha un solo figlio invece che due */
Value *UnaryExprAST::codegen(driver& drv) {
    Value *Operand = Val->codegen(drv);
    if (!Operand) 
        return nullptr;
    switch (Op) {
    case '!':
        return builder->CreateNot(Operand,"notres");
    default:  
        std::cout << Op << std::endl;
        return LogErrorV("Operatore unario non supportato");
    }
};

/********************* Call Expression Tree ***********************/
/* Call Expression Tree */
CallExprAST::CallExprAST(std::string Callee, std::vector<ExprAST*> Args):
    Callee(Callee),  Args(std::move(Args)) {};

lexval CallExprAST::getLexVal() const {
    lexval lval = Callee;
    return lval;
};

Value* CallExprAST::codegen(driver& drv) {
    // La generazione del codice corrispondente ad una chiamata di funzione
    // inizia cercando nel modulo corrente (l'unico, nel nostro caso) una funzione
    // il cui nome coincide con il nome memorizzato nel nodo dell'AST
    // Se la funzione non viene trovata (e dunque non è stata precedentemente definita)
    // viene generato un errore
    Function *CalleeF = module->getFunction(Callee);
    if (!CalleeF)
        return LogErrorV("Funzione non definita");
    // Il secondo controllo è che la funzione recuperata abbia tanti parametri
    // quanti sono gi argomenti previsti nel nodo AST
    if (CalleeF->arg_size() != Args.size())
        return LogErrorV("Numero di argomenti non corretto");
    // Passato con successo anche il secondo controllo, viene predisposta
    // ricorsivamente la valutazione degli argomenti presenti nella chiamata 
    // (si ricordi che gli argomenti possono essere espressioni arbitarie)
    // I risultati delle valutazioni degli argomenti (registri SSA, come sempre)
    // vengono inseriti in un vettore, dove "se li aspetta" il metodo CreateCall
    // del builder, che viene chiamato subito dopo per la generazione dell'istruzione
    // IR di chiamata
    std::vector<Value *> ArgsV;
    for (auto arg : Args) {
        ArgsV.push_back(arg->codegen(drv));
        if (!ArgsV.back())
            return nullptr;
    }
    return builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

/************************* If Expression Tree *************************/
IfExprAST::IfExprAST(ExprAST* Cond, ExprAST* TrueExp, ExprAST* FalseExp):
    Cond(Cond), TrueExp(TrueExp), FalseExp(FalseExp) {};
   
Value* IfExprAST::codegen(driver& drv) {
    // Viene dapprima generato il codice per valutare la condizione, che
    // memorizza il risultato (di tipo i1, dunque booleano) nel registro SSA 
    // che viene "memorizzato" in CondV. 
    Value* CondV = Cond->codegen(drv);
    if (!CondV)
        return nullptr;
    
    // Ora bisogna generare l'istruzione di salto condizionato, ma prima
    // vanno creati i corrispondenti basic block nella funzione attuale
    // (ovvero la funzione di cui fa parte il corrente blocco di inserimento)
    Function *function = builder->GetInsertBlock()->getParent();
    BasicBlock *TrueBB =  BasicBlock::Create(*context, "trueexp", function);
    // Il blocco TrueBB viene inserito nella funzione dopo il blocco corrente
    BasicBlock *FalseBB = BasicBlock::Create(*context, "falseexp");
    BasicBlock *MergeBB = BasicBlock::Create(*context, "endcond");
    // Gli altri due blocchi non vengono ancora inseriti perché le istruzioni
    // previste nel "ramo" true del condizionale potrebbe dare luogo alla creazione
    // di altri blocchi, che naturalmente andrebbero inseriti prima di FalseBB
    
    // Ora possiamo crere l'istruzione di salto condizionato
    builder->CreateCondBr(CondV, TrueBB, FalseBB);
    
    // "Posizioniamo" il builder all'inizio del blocco true, 
    // generiamo ricorsivamente il codice da eseguire in caso di
    // condizione vera e, in chiusura di blocco, generiamo il saldo 
    // incondizionato al blocco merge
    builder->SetInsertPoint(TrueBB);
    Value *TrueV = TrueExp->codegen(drv);
    if (!TrueV)
        return nullptr;
    builder->CreateBr(MergeBB);
    
    // Come già ricordato, la chiamata di codegen in TrueExp potrebbe aver inserito 
    // altri blocchi (nel caso in cui la parte trueexp sia a sua volta un condizionale).
    // Ne consegue che il blocco corrente potrebbe non coincidere più con TrueBB.
    // Il branch alla parte merge deve però essere effettuato dal blocco corrente,
    // che dunque va recuperato. Ed è fondamentale sapere da quale blocco origina
    // il salto perché tale informazione verrà utilizzata da un'istruzione PHI.
    // Nel caso in cui non sia stato inserito alcun nuovo blocco, la seguente
    // istruzione corrisponde ad una NO-OP
    TrueBB = builder->GetInsertBlock();
    function->insert(function->end(), FalseBB);
    
    // "Posizioniamo" il builder all'inizio del blocco false, 
    // generiamo ricorsivamente il codice da eseguire in caso di
    // condizione falsa e, in chiusura di blocco, generiamo il saldo 
    // incondizionato al blocco merge
    builder->SetInsertPoint(FalseBB);
    
    Value *FalseV = FalseExp->codegen(drv);
    if (!FalseV)
        return nullptr;
    builder->CreateBr(MergeBB);
    
    // Esattamente per la ragione spiegata sopra (ovvero il possibile inserimento
    // di nuovi blocchi da parte della chiamata di codegen in FalseExp), andiamo ora
    // a recuperare il blocco corrente 
    FalseBB = builder->GetInsertBlock();
    function->insert(function->end(), MergeBB);
    
    // Andiamo dunque a generare il codice per la parte dove i due "flussi"
    // di esecuzione si riuniscono. Impostiamo correttamente il builder
    builder->SetInsertPoint(MergeBB);

    // Il codice di riunione dei flussi è una "semplice" istruzione PHI: 
    //a seconda del blocco da cui arriva il flusso, TrueBB o FalseBB, il valore
    // del costrutto condizionale (si ricordi che si tratta di un "expression if")
    // deve essere copiato (in un nuovo registro SSA) da TrueV o da FalseV
    // La creazione di un'istruzione PHI avviene però in due passi, in quanto
    // il numero di "flussi entranti" non è fissato.
    // 1) Dapprima si crea il nodo PHI specificando quanti sono i possibili nodi sorgente
    // 2) Per ogni possibile nodo sorgente, viene poi inserita l'etichetta e il registro
    //    SSA da cui prelevare il valore 

    PHINode *PN = builder->CreatePHI(Type::getDoubleTy(*context), 2, "condval");
    PN->addIncoming(TrueV, TrueBB);
    PN->addIncoming(FalseV, FalseBB);
    return PN;
};

/************************* If Statement Tree *************************/
IfStmtAST::IfStmtAST(ExprAST* Cond, StmtAST* TrueStmt, StmtAST* FalseStmt):
    Cond(Cond), TrueStmt(TrueStmt), FalseStmt(FalseStmt) {};

Value* IfStmtAST::codegen(driver& drv) {
    // Viene dapprima generato il codice per valutare la condizione, che
    // memorizza il risultato (di tipo i1, dunque booleano) nel registro SSA 
    // che viene "memorizzato" in CondV. 
    Value* CondV = Cond->codegen(drv);
    if (!CondV)
        return nullptr;
    
    // Ora bisogna generare l'istruzione di salto condizionato, ma prima
    // vanno creati i corrispondenti basic block nella funzione attuale
    // (ovvero la funzione di cui fa parte il corrente blocco di inserimento)
    Function *function = builder->GetInsertBlock()->getParent();
    BasicBlock *TrueBB =  BasicBlock::Create(*context, "trueexp", function);
    // Il blocco TrueBB viene inserito nella funzione dopo il blocco corrente

    //il blocco false viene generato solo se è effettivamente presente il ramo false dell'if
    BasicBlock *FalseBB;
    if(FalseStmt)
        FalseBB = BasicBlock::Create(*context, "falseexp");

    BasicBlock *MergeBB = BasicBlock::Create(*context, "endcond");
    // Gli altri due blocchi non vengono ancora inseriti perché le istruzioni
    // previste nel "ramo" true del condizionale potrebbe dare luogo alla creazione
    // di altri blocchi, che naturalmente andrebbero inseriti prima di FalseBB
    
    // L'istruzione di salto condizionato tiene conto dell'eventuale assenza del ramo false
    if(FalseStmt)
        builder->CreateCondBr(CondV, TrueBB, FalseBB);
    else
        // se il false non è presente le due strade possibili saranno true e merge
        builder->CreateCondBr(CondV, TrueBB, MergeBB);
    
    // "Posizioniamo" il builder all'inizio del blocco true, 
    // generiamo ricorsivamente il codice da eseguire in caso di
    // condizione vera e, in chiusura di blocco, generiamo il saldo 
    // incondizionato al blocco merge
    builder->SetInsertPoint(TrueBB);
    Value *TrueV = TrueStmt->codegen(drv);
    if (!TrueV)
        return nullptr;
    builder->CreateBr(MergeBB);
    
    // Come già ricordato, la chiamata di codegen in TrueExp potrebbe aver inserito 
    // altri blocchi (nel caso in cui la parte trueexp sia a sua volta un condizionale).
    // Ne consegue che il blocco corrente potrebbe non coincidere più con TrueBB.
    // Il branch alla parte merge deve però essere effettuato dal blocco corrente,
    // che dunque va recuperato. Ed è fondamentale sapere da quale blocco origina
    // il salto perché tale informazione verrà utilizzata da un'istruzione PHI.
    // Nel caso in cui non sia stato inserito alcun nuovo blocco, la seguente
    // istruzione corrisponde ad una NO-OP
    TrueBB = builder->GetInsertBlock();

    //se il ramo false non dovesse essere presente ovviamente non viene generato codice
    Value *FalseV;
    if(FalseStmt){
        function->insert(function->end(), FalseBB);
        
        // "Posizioniamo" il builder all'inizio del blocco false, 
        // generiamo ricorsivamente il codice da eseguire in caso di
        // condizione falsa e, in chiusura di blocco, generiamo il saldo 
        // incondizionato al blocco merge
        builder->SetInsertPoint(FalseBB);
        
        FalseV = FalseStmt->codegen(drv);
        if (!FalseV)
            return nullptr;
        builder->CreateBr(MergeBB);
        
        // Esattamente per la ragione spiegata sopra (ovvero il possibile inserimento
        // di nuovi blocchi da parte della chiamata di codegen in FalseExp), andiamo ora
        // a recuperare il blocco corrente 
        FalseBB = builder->GetInsertBlock();
    }

    function->insert(function->end(), MergeBB);
    
    // Andiamo dunque a generare il codice per la parte dove i due "flussi"
    // di esecuzione si riuniscono. Impostiamo correttamente il builder
    builder->SetInsertPoint(MergeBB);

    

    /*  Il codice di riunione dei flussi è leggermente diverso a quello di IfExpr dato che occorre
        generare un nodo PHI solamente nel caso in cui ci sia un ramo false, altrimenti il valore finale
        del costrutto sarà sicuramente quello calcolato nel ramo true */
    if(FalseStmt){
        PHINode *PN = builder->CreatePHI(Type::getDoubleTy(*context), 2, "condval");
        PN->addIncoming(TrueV, TrueBB);
        PN->addIncoming(FalseV, FalseBB);
        return PN;
    }
    else{
        return TrueV;
    }
};

/************************* For Statement Tree *************************/

ForStmtAST::ForStmtAST(InitAST* Init, ExprAST* Cond, RootAST* Assignment, StmtAST* Stmt):
    Init(Init), Cond(Cond), Assignment(Assignment), Stmt(Stmt) {};

Value* ForStmtAST::codegen(driver& drv) {
    /*  L'idea è quella di dividere la generazione del codice per il for in 4 sezioni:
        1) assegnamento/definizione variabile su cui iterare, gestito tramite la classe Init (inizializzazione)
        2) blocco condizione -> se condizione vera si entra nel loop altrimenti si esce
        3) blocco loop -> vengono eseguite le istruzioni e si salta incondizionatamente al blocco condizione alla fine
        4) blocco endloop -> il flusso del programma dovrà riprendere da questo punto una volta terminato il for */

    /*  Creazione blocco di endloop (dove bisognerà saltare una volta terminato il loop). Questo blocco resta 
        sconnesso momentaneamente perchè potrebbero essere generati altri blocchi prima */
    BasicBlock *MergeBB = BasicBlock::Create(*context, "endloop");

    /*** 1) ASSEGNAMENTO/DEFINIZIONE ***/

    //questa map viene utilizzata nel caso ci sia una definizione per oscurare eventuali variabili ononime
    std::map<std::string, AllocaInst*> AllocaTmp;
    if(!Init->isDefinition()){
        //generazione codice assegnamento variabile, non rimane nient'altro da fare
        if(!Init->codegen(drv)) return nullptr;
    }
    else{
        /*  Generazione codice di definizione variabile, in questo caso la codegen restituisce un'AllocaInst invece che un Value
            che è una classe madre. Per far funzionare il meccanismo di offuscamento è necessario fare un cast da Value ad AllocaInst
            in questo modo è possibile inserire temporaneamente il valore sel registro SSA all'interno della NamedValues */
        AllocaInst *initVal = dyn_cast<AllocaInst>(Init->codegen(drv));
        if(!initVal) return nullptr;

        AllocaTmp[Init->getName()] = drv.NamedValues[Init->getName()];
        drv.NamedValues[Init->getName()] = initVal;
    }

    /*** 2) VALUTAZIONE CONDIZIONE ***/

    /*  Creazione blocco per valutare la condizione, si salterà a questo blocco alla fine di ogni iterazione
        E' il primo blocco ad essere eseguito quindi viene agganciato alla fine della funzione e posiziono il builder
        al suo interno */
    Function *function = builder->GetInsertBlock()->getParent();
    BasicBlock *CondBB =  BasicBlock::Create(*context, "condeval", function);

    /*  Una volta generato il blocco viene inserito un salto incondizionato proveniente dal blocco precedente per
        permettere di "entrare" concettualmente nella fase di esecuzione vera e propria del for */
    builder->CreateBr(CondBB);

    //ora si posiziona il builder all'interno del blocco di valutazione della condizione
    builder->SetInsertPoint(CondBB);

    /*  Creazione blocco loop, viene lasciato sconnesso momentaneamente in attesa che venga generato il codice
        per la valutazione della condizione */
    BasicBlock *ForBB =  BasicBlock::Create(*context, "forloop");

    //generazione codice per valutazione condizione (la condizione viene valutata all'inizio di ogni iterazione)
    Value* CondV = Cond->codegen(drv);
    if (!CondV)
        return nullptr;

    //generazione codice per salto condizionato
    builder->CreateCondBr(CondV, ForBB, MergeBB);
    
    /*** 3) LOOP ***/
    
    //viene agganciato il blocco che gestisce il loop creato precedentemente
    function->insert(function->end(), ForBB);
    builder->SetInsertPoint(ForBB);

    //generazione codice del body interno al loop e return nullptr se ci sono stati errori
    if(!Stmt->codegen(drv))
        return nullptr;
    
    //serve a recuperare il blocco corrente nel caso siano stati generati altri blocchi da stmt->codegen
    builder->GetInsertBlock();
    
    //generazione codice dell'assegnamento finale e return nullptr se ci sono stati errori
    Value *var = Assignment->codegen(drv);
    if(!var)
        return nullptr;

    //salto incondizionato a blocco Cond per effettuare una nuova verifica della condizione del loop
    builder->CreateBr(CondBB);

    /*** 4) EXIT LOOP ***/

    //viene agganciato il blocco di uscita dal loop e si posiziona il builder al suo interno
    function->insert(function->end(), MergeBB);
    builder->SetInsertPoint(MergeBB);

    // viene reinserita nella NamedValues la variabile che prima era stata offuscata da un'eventuale definizione

    if(Init->isDefinition())
        drv.NamedValues[Init->getName()] = AllocaTmp[Init->getName()];

    // viene restituita una costante pari a zero per indicare che non c'è stato nessun errore
    return ConstantFP::get(*context, APFloat(0.0));
}

/********************** Block Expression Tree *********************/
BlockAST::BlockAST(std::vector<BindingAST*> Def, std::vector<StmtAST*> Stmts): 
         Def(std::move(Def)), Stmts(std::move(Stmts)) {};

Value* BlockAST::codegen(driver& drv) {
    // Un blocco è un insieme di statement preceduto dalla definizione di una o più variabili locali.
    // Le definizioni sono opzionali e tuttavia necessarie perché l'uso di un blocco
    // abbia senso. Ad ogni variabile deve essere associato il valore di una costante o il valore di
    // un'espressione. Nell'espressione, arbitraria, possono chiaramente comparire simboli di
    // variabile. Al riguardo, la gestione dello scope (ovvero delle regole di visibilità)
    // è implementata nel modo seguente, in cui, come esempio, consideremo la definzione: var y = x+1
    // 1) Viene dapprima generato il codice per valutare l'espressione x+1.
    //    L'area di memoria da cui "prelevare" il valore di x è scritta in un
    //    registro SSA che è parte della (rappresentazione interna della) istruzione alloca usata
    //    per allocare la memoria corrispondente e che è registrata nella symbol table
    //    Per i parametri della funzione, l'istruzione di allocazione viene generata (come già sappiamo)
    //    dalla chiamata di codegen in FunctionAST. Per le variabili locali viene generata nel presente
    //    contesto. Si noti, di passaggio, che tutte le istruzioni di allocazione verranno poi emesse
    //    nell'entry block, in ordine cronologico rovesciato (rispetto alla generazione). Questo perché
    //    la routine di utilità (CreateEntryBlockAlloca) genera sempre all'inizio del blocco.
    // 2) Ritornando all'esempio, bisogna ora gestire l'assegnamento ad y gestendone la visibilità. 
    //    Come prima cosa viene generata l'istruzione alloca per y. 
    //    Questa deve essere inserita nella symbol table per futuri riferimenti ad y
    //    all'interno del blocco. Tuttavia, se un'istruzione alloca per y fosse già presente nella symbol
    //    table (nel caso y sia un parametro) bisognerebbe "rimuoverla" temporaneamente e re-inserirla
    //    all'uscita del blocco. Questo è ciò che viene fatto dal presente codice, che utilizza
    //    al riguardo il vettore di appoggio "AllocaTmp" (che naturalmente è un vettore di
    //    di (puntatori ad) istruzioni di allocazione

    
    std::vector<AllocaInst*> AllocaTmp;
    for (int i=0, e=Def.size(); i<e; i++) {
        // Per ogni definizione di variabile si genera il corrispondente codice che
        // (in questo caso) non restituisce un registro SSA ma l'istruzione di allocazione
        AllocaInst *boundval = Def[i]->codegen(drv);
        if (!boundval) 
            return nullptr;
        // Viene temporaneamente rimossa la precedente istruzione di allocazione
        // della stessa variabile (nome) e inserita quella corrente
        AllocaTmp.push_back(drv.NamedValues[Def[i]->getName()]);
        drv.NamedValues[Def[i]->getName()] = boundval;
    };

    /*  Generazione del codice relativo ai vari statement, ad ogni generazione viene controllato
        se è stato restituito un puntatore nullo in modo da segnalare eventuali errori  */
    Value *blockvalue;
    for(int i=0, s=Stmts.size(); i<s; i++){
        blockvalue = Stmts[i]->codegen(drv);
        if(!blockvalue) return nullptr;
    }
        
    // Prima di uscire dal blocco, si ripristina lo scope esterno al costrutto
    for (int i=0, e=Def.size(); i<e; i++) {
            drv.NamedValues[Def[i]->getName()] = AllocaTmp[i];
    };
    
    /*  Il valore di ritorno del blocco corrisponde all'ultimo registro SSA definito da uno statement */
    return blockvalue;
};

/************************* Init Tree *************************/

InitAST::InitAST(const std::string Name, bool definition):
    Name(Name), definition(definition) {};

bool InitAST::isDefinition(){
    return definition;
}

const std::string& InitAST::getName() const { 
   return Name; 
};

/************************* Binding Tree *************************/

BindingAST::BindingAST(const std::string Name):
    InitAST(Name, true) {};

/************************* Var binding Tree *************************/
VarBindingAST::VarBindingAST(const std::string Name, ExprAST* Val):
    BindingAST(Name), Val(Val) {};

AllocaInst* VarBindingAST::codegen(driver& drv) {
    
    // Viene subito recuperato il riferimento alla funzione in cui si trova
    // il blocco corrente. Il riferimento è necessario perché lo spazio necessario
    // per memorizzare una variabile (ovunque essa sia definita, si tratti cioè
    // di un parametro oppure di una variabile locale ad un blocco espressione)
    // viene sempre riservato nell'entry block della funzione. Ricordiamo che
    // l'allocazione viene fatta tramite l'utility CreateEntryBlockAlloca
    Function *fun = builder->GetInsertBlock()->getParent();
    // Ora viene generato il codice che definisce il valore della variabile
    Value *BoundVal = Val->codegen(drv);
    if (!BoundVal)  // Qualcosa è andato storto nella generazione del codice?
        return nullptr;
    // Se tutto ok, si genera l'struzione che alloca memoria per la varibile ...
    AllocaInst *Alloca = CreateEntryBlockAlloca(fun, Name);
    // ... e si genera l'istruzione per memorizzarvi il valore dell'espressione,
    // ovvero il contenuto del registro BoundVal
    builder->CreateStore(BoundVal, Alloca);
    
    // L'istruzione di allocazione (che include il registro "puntatore" all'area di memoria
    // allocata) viene restituita per essere inserita nella symbol table
    return Alloca;
};

/************************* Array binding Tree *************************/
ArrayBindingAST::ArrayBindingAST(const std::string Name, double itemNum, std::vector<ExprAST*> initExprList):
    BindingAST(Name), itemNum(itemNum), initExprList(std::move(initExprList)) {};

AllocaInst* ArrayBindingAST::codegen(driver& drv) {
    
    //recupero riferimento alla funzione corrente per inserimento istruzione alloca ad inizio blocco
    Function *fun = builder->GetInsertBlock()->getParent();
 
    //necessario per definire tipo dell'array e numero di "celle" contigue da allocare
    ArrayType *AT = ArrayType::get(Type::getDoubleTy(*context),itemNum);
    //viene allocata effettivamente memoria per l'array
    AllocaInst *Alloca = CreateEntryBlockAlloca(fun, Name, AT);

    /*  Controllo numero elementi eventualmente inizializzati, l'importante è che non siano maggiori del
        numero di elementi dell'array (itemNum) ma ho scelto di rendere possibile l'inizializzazione solo
        se il numero di elementi inizializzati è pari a itemNum o se è pari a zero*/
    if(itemNum != initExprList.size() && initExprList.size()!=0){
        LogErrorV("Array con " + std::to_string(itemNum) + " elementi inizializzato con " + std::to_string(initExprList.size()) + " espressioni");
        return nullptr;
    }

    for(int i = 0; i < initExprList.size(); i++){
        //generazione codice espressione corrente
        Value *exprRes = initExprList[i]->codegen(drv);
        //in caso ci fosse qualche problema durante la generazione del codice dell'espressione
        if(!exprRes) 
            return nullptr;
        //creazione costante intera da utilizzare per accedere alla cella dell'array
        Value *index = ConstantInt::get(*context, APInt(32, i, true));
        //accesso alla cella dell'array
        Value *cell = builder->CreateInBoundsGEP(AT, Alloca, {ConstantInt::get(Type::getInt32Ty(*context), 0), index});
        //semplice store del valore dell'espressione all'interno della cella
        builder->CreateStore(exprRes, cell);
    }
    
    // L'istruzione di allocazione (che include il registro "puntatore" all'area di memoria
    // allocata) viene restituita per essere inserita nella symbol table
    return Alloca;
};

/************************* Assignment Tree *************************/

AssignmentAST::AssignmentAST(const std::string Name):
    InitAST(Name, false) {};

/************************** Var Assignment Tree *************************/
VarAssignmentAST::VarAssignmentAST(const std::string Name, ExprAST* Val):
   AssignmentAST(Name), Val(Val) {};

Value* VarAssignmentAST::codegen(driver& drv) {
    //priorità a variabili locali, se un nome non è definito tra le variabili locali allora si passa 
    //a controllare le variabili globali
    Value *var = drv.NamedValues[Name];
    Type *varType;
    if (var){
        varType = drv.NamedValues[Name]->getAllocatedType();
    }
    else{
        GlobalVariable *globalVar = module->getNamedGlobal(Name);
        if(!globalVar)
            return LogErrorV("Variabile "+Name+" non definita");
        varType = globalVar->getValueType();
        var = globalVar;       
    }

    //controllo per verificare che la variabile non sia in realtà un array
    if(varType->isArrayTy())
        return LogErrorV("Array utilizzato senza operatore []");

    
    // Ora viene generato il codice che definisce il valore della variabile
    Value *BoundVal = Val->codegen(drv);
    if (!BoundVal)  // Qualcosa è andato storto nella generazione del codice?
        return nullptr;
    // ... e si genera l'istruzione per memorizzarvi il valore dell'espressione,
    // ovvero il contenuto del registro BoundVal
    builder->CreateStore(BoundVal, var);

    /*  Viene restituito un registro SSA che contiene il valore del registro (o della variabile globale) in cui è stata fatta
        la store. Questo viene fatto per ritornare sempre un valore double (dato che altrimenti, ritornando direttamente
        il registro o la variabile globale si avrebbe un valore di tipo puntatore). Tutte le funzioni si aspettano un tipo di ritorno double
        quindi questa scelta si rende necessaria */
    return builder->CreateLoad(varType, var, ("ret_" + Name).c_str());
};

/************************** Array Assignment Tree *************************/
ArrayAssignmentAST::ArrayAssignmentAST(const std::string Name, ExprAST* Idx, ExprAST* Val):
   AssignmentAST(Name), Idx(Idx), Val(Val) {};

Value* ArrayAssignmentAST::codegen(driver& drv) {

    //priorità a variabili locali, se un nome non è definito tra le variabili locali allora si passa 
    //a controllare le variabili globali
    Type *varType;
    Value *array = drv.NamedValues[Name];
    if(array){
        varType = drv.NamedValues[Name]->getAllocatedType();
    }
    else{
        GlobalVariable* globalArray = module->getNamedGlobal(Name);
        if(globalArray){
            varType = globalArray->getValueType();
            array = globalArray;
        }
        else{
            return LogErrorV("Array "+Name+" non definito");  
        }
    }

    //controllo per verificare che la variabile sia effettivamente un array
    if(!varType->isArrayTy())
        return LogErrorV("Array utilizzato senza operatore []");

    //generazione codice che definisce il valore dell'indice
    Value *indexVal = Idx->codegen(drv);
    if (!indexVal)  // Qualcosa è andato storto nella generazione del codice?
        return nullptr;

    //il valore prodotto dalla codegen dell'indice deve essere convertito da double ad intero
    indexVal = builder->CreateFPToSI(indexVal, Type::getInt32Ty(*context), "intindex");

    // Ora viene generato il codice che definisce il valore della variabile
    Value *BoundVal = Val->codegen(drv);
    if (!BoundVal)  // Qualcosa è andato storto nella generazione del codice?
        return nullptr;

    //recupero tipo array allocato per accedere effettivamente alla cella
    ArrayType *AT = dyn_cast<ArrayType>(varType);
    //accesso effettivo alla cella dell'array
    Value *cell = builder->CreateInBoundsGEP(AT, array, {ConstantInt::get(Type::getInt32Ty(*context), 0), indexVal});
    //semplice store del valore dell'espressione all'interno della cella
    builder->CreateStore(BoundVal, cell);

    /*  Viene restituito un registro SSA che contiene il valore della cella dell'array in cui è stata eseguita
        la store. Questo viene fatto per ritornare sempre un valore double (dato che altrimenti, ritornando direttamente
        la cella si avrebbe un valore di tipo puntatore). Tutte le funzioni si aspettano un tipo di ritorno double
        quindi questa scelta si rende necessaria */
    return builder->CreateLoad(AT->getElementType(), cell, ("retv_" + Name).c_str());
};

/************************* Prototype Tree *************************/
PrototypeAST::PrototypeAST(std::string Name, std::vector<std::string> Args):
    Name(Name), Args(std::move(Args)), emitcode(true) {};  //Di regola il codice viene emesso

lexval PrototypeAST::getLexVal() const {
    lexval lval = Name;
    return lval;	
};

const std::vector<std::string>& PrototypeAST::getArgs() const { 
    return Args;
};

// Previene la doppia emissione del codice. Si veda il commento più avanti.
void PrototypeAST::noemit() { 
    emitcode = false; 
};

Function *PrototypeAST::codegen(driver& drv) {
    // Costruisce una struttura, qui chiamata FT, che rappresenta il "tipo" di una
    // funzione. Con ciò si intende a sua volta una coppia composta dal tipo
    // del risultato (valore di ritorno) e da un vettore che contiene il tipo di tutti
    // i parametri. Si ricordi, tuttavia, che nel nostro caso l'unico tipo è double.
    
    // Prima definiamo il vettore (qui chiamato Doubles) con il tipo degli argomenti
    std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(*context));
    // Quindi definiamo il tipo (FT) della funzione
    FunctionType *FT = FunctionType::get(Type::getDoubleTy(*context), Doubles, false);
    // Infine definiamo una funzione (al momento senza body) del tipo creato e con il nome
    // presente nel nodo AST. ExternalLinkage vuol dire che la funzione può avere
    // visibilità anche al di fuori del modulo
    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, *module);

    // Ad ogni parametro della funzione F (che, è bene ricordare, è la rappresentazione 
    // llvm di una funzione, non è una funzione C++) attribuiamo ora il nome specificato dal
    // programmatore e presente nel nodo AST relativo al prototipo
    unsigned Idx = 0;
    for (auto &Arg : F->args())
        Arg.setName(Args[Idx++]);

    /* Abbiamo completato la creazione del codice del prototipo.
        Il codice può quindi essere emesso, ma solo se esso corrisponde
        ad una dichiarazione extern. Se invece il prototipo fa parte
        della definizione "completa" di una funzione (prototipo+body) allora
        l'emissione viene fatta al momendo dell'emissione della funzione.
        In caso contrario nel codice si avrebbe sia una dichiarazione
        (come nel caso di funzione esterna) sia una definizione della stessa
        funzione.
    */
    
    if (emitcode) {
        F->print(errs());
        fprintf(stderr, "\n");
    };
    
    return F;
}

/************************* Function Tree **************************/
FunctionAST::FunctionAST(PrototypeAST* Proto, BlockAST* Body): Proto(Proto), Body(Body) {};

Function *FunctionAST::codegen(driver& drv) {
    // Verifica che la funzione non sia già presente nel modulo, cioò che non
    // si tenti una "doppia definizion"
    Function *function = 
        module->getFunction(std::get<std::string>(Proto->getLexVal()));
    // Se la funzione non è già presente, si prova a definirla, innanzitutto
    // generando (ma non emettendo) il codice del prototipo
    if (!function)
        function = Proto->codegen(drv);
    else
        return nullptr;
    // Se, per qualche ragione, la definizione "fallisce" si restituisce nullptr
    if (!function)
        return nullptr;  

    // Altrimenti si crea un blocco di base in cui iniziare a inserire il codice
    BasicBlock *BB = BasicBlock::Create(*context, "entry", function);
    builder->SetInsertPoint(BB);
    
    // Ora viene la parte "più delicata". Per ogni parametro formale della
    // funzione, nella symbol table si registra una coppia in cui la chiave
    // è il nome del parametro mentre il valore è un'istruzione alloca, generata
    // invocando l'utility CreateEntryBlockAlloca già commentata.
    // Vale comunque la pena ricordare: l'istruzione di allocazione riserva 
    // spazio in memoria (nel nostro caso per un double) e scrive l'indirizzo
    // in un registro SSA
    // Il builder crea poi un'istruzione che memorizza il valore del parametro x
    // (al momento contenuto nel registro SSA %x) nell'area di memoria allocata.
    // Si noti che il builder conosce il registro che contiene il puntatore all'area
    // perché esso è parte della rappresentazione C++ dell'istruzione di allocazione
    // (variabile Alloca) 
    
    for (auto &Arg : function->args()) {
        // Genera l'istruzione di allocazione per il parametro corrente
        AllocaInst *Alloca = CreateEntryBlockAlloca(function, Arg.getName());
        // Genera un'istruzione per la memorizzazione del parametro nell'area
        // di memoria allocata
        builder->CreateStore(&Arg, Alloca);
        // Registra gli argomenti nella symbol table per eventuale riferimento futuro
        drv.NamedValues[std::string(Arg.getName())] = Alloca;
    } 
    
    // Ora può essere generato il codice corssipondente al body (che potrà
    // fare riferimento alla symbol table)
    if (Value *RetVal = Body->codegen(drv)) {
        // Se la generazione termina senza errori, ciò che rimane da fare è
        // di generare l'istruzione return, che ("a tempo di esecuzione") prenderà
        // il valore lasciato nel registro RetVal

        builder->CreateRet(RetVal);

        // Effettua la validazione del codice e un controllo di consistenza
        verifyFunction(*function);
    
        // Emissione del codice su su stderr) 
        function->print(errs());
        fprintf(stderr, "\n");
        return function;
    }

    // Errore nella definizione. La funzione viene rimossa
    function->eraseFromParent();
    return nullptr;
};

