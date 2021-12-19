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
            std::vector<BasicBlock*> bbs;
        
        public:
            Node();
            ~Node();
            //TO.DO.: Provide API for setting succ & pred(including clear, push_back, return specified node)
            void clear_succ(Node *);
            void clear_pred(Node *);
            void add_succ(Node*);
            void add_pred(Node*);
            size_t get_succ_num();
            size_t get_pred_num();
            Node * get_succ(size_t);
            Node * get_pred(size_t);
            virtual void dump_func();
            virtual void dump_inst();
            void add_input_value(Value *);
            void add_output_value(Value *);
            std::vector<Value *> get_input_value();
            std::vector<Value *> get_output_value();
            virtual CallInst * getCallInst();
            virtual Function * getFunction();
            virtual void CollectBBs(Module &);
            void dumpBBs();
            void addBB(BasicBlock*);
            void deleteBB(BasicBlock*);
            std::vector<BasicBlock*> getBBs();
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
            CallInst * getCallInst();
            Function * getFunction();
            virtual void CollectBBs(Module &);
            virtual void SetStream(GlobalValue *);
    };

    class GPUFuncNode : public FuncNode{
        public:
            GPUFuncNode(CallInst *, Function *, bool, bool);
            void CollectBBs(Module &);
            void SetStream(GlobalValue *);
    };

    class CPUFuncNode : public FuncNode{
        public:
            CPUFuncNode(CallInst *, Function *, bool, bool);
    };

    class MemcpyNode : public FuncNode{
        public:
            MemcpyNode(CallInst *, Function *, bool,bool);
            void CollectBBs(Module &);
            void SetStream(GlobalValue *);
    };

    class SuperNode{
        private:
            Node * node_list;                   //Contains one function node and its pre inst node
    };

    class Seq_Graph : public Node{
        private:
            Node * Entry_Node;
            Node * End_Node;

        public:
            Seq_Graph();
            void Insert(Node *, Node *);                //First is inserted node, Second is the node before inserted node. In SeqGraph, this may not be neccessary
            void Delete(Node *);
            Node * get_last_Node();
            bool IsEndNode(Node*);
            Node * getEntryNode();
            void WalkGraph();                           //TO.DO.: Pass a function pointer which is applied on each element of graph
            void Print_allNode();
    };

    //A function has a graph
    //A branch has a graph(if-else), including loop
    class DAG : public Node{
        private:
            Node * Entry_Node;                  //Its next nodes should be those nodes with no undone 
            Node * End_Node;
            size_t max_width;
            size_t n_level;
            std::unordered_map<size_t,std::vector<Node*>> level_nodes_map;        //Partitioned by level
            std::unordered_map<Node*,size_t> node_level_map;                //identify level of one node
            std::unordered_map<Node*,std::vector<Node*>> pred_map;
        public:
            DAG();
            void ConstructFromSeq_Graph(Seq_Graph *);
            void Insert(Node *);                                            //Need to find all pred nodes of inserted node,       
            void Delete(Node *);
            std::vector<Node*> get_level_nodes(size_t);
            size_t get_level(Node *);
            void levelize();
            void dump_level();
            Node * reverse_find_pred(Value *, bool);                              //backward BFS
            void dump();                                                    //Generate a graph
    };
}
    