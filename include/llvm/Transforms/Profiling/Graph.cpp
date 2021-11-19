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
#include "Graph.h"

using namespace llvm;

namespace graph{

    Node::Node()
    {
        undone_pred = 0; 
        succ.clear(); 
        pred.clear();
    }

    InstNode::InstNode(Instruction * target_inst)
    {
        inst = target_inst;
    }

    FuncNode::FuncNode(CallInst * ci, Function * cf, bool flag = false)
    {
        call_inst = ci;
        called_func = cf;
        gpu_flag = flag;
    }

    GPUFuncNode::GPUFuncNode(CallInst * ci, Function * cf, bool flag = true):FuncNode(ci,cf,flag)
    {
        //do nothing?
    }

    CPUFuncNode::CPUFuncNode(CallInst * ci, Function * cf, bool flag = false):FuncNode(ci,cf,flag)
    {
        //do nothing?
    }

    MemcpyNode::MemcpyNode(CallInst * ci, Function * cf, bool flag = false):FuncNode(ci,cf,flag)
    {
        //do nothing?
    }

    GRAPH::GRAPH():Node()
    {
        Entry_Node = new Node();
        End_Node = new Node();
        Entry_Node->pred.clear();
        Entry_Node->succ.clear();
        End_Node->succ.clear();
        End_Node->undone_pred++;
        End_Node->pred.clear();
    }

    
}