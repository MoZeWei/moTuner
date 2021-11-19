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
#include<unordered_map>

using namespace llvm;

namespace graph{
    struct Node{
        std::vector<Node*> pred;
        size_t undone_pred;
        std::vector<Node*> succ;
        Node();
    };

    struct InstNode : Node{
        Instruction * inst;
        InstNode(Instruction*);
    };

    struct FuncNode : Node{
        CallInst * call_inst;
        Function * called_func;
        bool gpu_flag;
        FuncNode(CallInst*, Function *, bool);
        void SetupDep();
    };

    struct GPUFuncNode : FuncNode{
        GPUFuncNode(CallInst *, Function *, bool);
    };

    struct CPUFuncNode : FuncNode{
        CPUFuncNode(CallInst *, Function *, bool);
    };

    struct MemcpyNode : FuncNode{
        MemcpyNode(CallInst *, Function *, bool);
    };

    struct SuperNode{
        Node * node_list;                   //Contains one function node and its pre inst node
    };

    //A function has a graph
    //A branch has a graph(if-else), including loop
    struct GRAPH : Node{
        Node * Entry_Node;                  //Its next nodes should be those nodes with no undone 
        Node * End_Node;
        GRAPH();
        void Insert(Node *, std::vector<Node*> );        //insert a new node after pred nodes           
        void Delete(Node *);
    };
}
    