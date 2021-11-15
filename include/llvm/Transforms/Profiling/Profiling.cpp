#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"

using namespace llvm;

//This pass is to insert timing function before & after each function and also analysis 
//data dependencies among functions.

//Construct graph for each function

namespace{
    struct Node{
        Function * func;
        std::vector<Node*> succ;
        size_t undone_prec;
    };

    struct SuperNode{
        Node * node_list;
    };

    //A function has a graph
    //A branch has a graph(if-else), including loop
    struct Graph{
        Node * Entry_Node;

    };


    struct Profiling : public ModulePass{
        static char ID;

        Profiling();
        ~Profiling();

        bool runOnModule(Module & M) override;

        //data
        std::string profiling_data_file_path;


    };

    Profiling::Profiling() : ModulePass(ID){

    }

}

char Profiling::ID = 0;

static RegisterPass<Profiling> X("profiling","A Pass to Profile and run data analysis",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new Profiling());});













