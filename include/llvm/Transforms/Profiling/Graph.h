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
    class Node{
        private:
            std::vector<Node*> pred;
            size_t undone_pred;
            std::vector<Node*> succ;
            std::vector<Value*> input_value;
            std::vector<Value*> output_value;
        
        public:
            Node();
            ~Node();
            //TO.DO.: Provide API for setting succ & pred(including clear, push_back, return specified node)
            void clear_succ(Node *);
            void clear_pred(Node *);
            void add_succ(Node*);
            void add_pred(Node*);
            Node * get_succ(size_t);
            Node * get_pred(size_t);
            virtual void dump_func();
            virtual void dump_inst();
            virtual void add_input_value(Value *);
            virtual void add_output_value(Value *);
            virtual std::vector<Value *> get_input_value();
            virtual std::vector<Value *> get_output_value();
    };

    class InstNode : public Node{
        private:
        Instruction * inst;
        
        public:
            InstNode(Instruction*);
            ~InstNode();
    };

    class FuncNode : public Node{
        private:
            CallInst * call_inst;
            Function * called_func;
            bool gpu_flag;                                  //This is to distinguish whether a function is a kernel/memcpy or cpu function
            bool has_graph;                                 //This is to distinguish whether a function is library_func or user-defined function

        public:
            FuncNode(CallInst*, Function *, bool, bool);
            void SetupDep();                                //TO.DO.
            std::string get_func_name();
            void dump_func();
            void dump_inst();
    };

    class GPUFuncNode : public FuncNode{
        public:
            GPUFuncNode(CallInst *, Function *, bool, bool);
    };

    class CPUFuncNode : public FuncNode{
        public:
            CPUFuncNode(CallInst *, Function *, bool, bool);
    };

    class MemcpyNode : public FuncNode{
        public:
            MemcpyNode(CallInst *, Function *, bool,bool);
    };

    class SuperNode{
        private:
            Node * node_list;                   //Contains one function node and its pre inst node
    };

    //A function has a graph
    //A branch has a graph(if-else), including loop
    class DAG : public Node{
        private:
            Node * Entry_Node;                  //Its next nodes should be those nodes with no undone 
            Node * End_Node;
        public:
            DAG();
            void Insert(Node *, std::vector<Node*> );        //insert a new node after pred nodes           
            void Delete(Node *);

    };

    class Seq_Graph : public Node{
        private:
            Node * Entry_Node;
            Node * End_Node;

        public:
            Seq_Graph();
            void Insert(Node *, Node *);
            void Delete(Node *);
            Node * get_last_Node();
            bool IsEndNode(Node*);
            Node * getEntryNode();
            void WalkGraph();                           //TO.DO.: Pass a function pointer which is applied on each element of graph
            void Print_allNode();
    };
}
    