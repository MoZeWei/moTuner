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

    Node::~Node()
    {
        //do nothing
    }

    void Node::clear_succ(Node * target = nullptr)
    {
        if(target == nullptr)   succ.clear();
        else
        {
            size_t i = 0;
            for(; i < succ.size(); i++)
            {
                if(target == succ[i]) break;
            }
            if(i >= succ.size())
            {
                std::cout<<"Cannot clear target succ node"<<std::endl;
                exit(1);
            }
            else
            {
                succ.erase(succ.begin() + i);
            }
        }
    }

    void Node::clear_pred(Node * target = nullptr)
    {
        if(target == nullptr)
        {
            pred.clear();
            undone_pred = 0;
        }
        else
        {
            size_t i = 0;
            for(; i < pred.size(); i++)
            {
                if(target == pred[i]) break;
            }
            if(i >= pred.size())
            {
                std::cout<<"Cannot clear target pred node"<<std::endl;
                exit(1);
            }
            else
            {
                pred.erase(pred.begin() + i);
            }
        }
    }

    void Node::add_succ(Node * node)
    {
        succ.push_back(node);
    }

    void Node::add_pred(Node * node)
    {
        pred.push_back(node);
        undone_pred++;
    }

    Node * Node::get_pred(size_t i)
    {
        if(i > pred.size()) return nullptr;
        else return pred[i];
    }

    Node * Node::get_succ(size_t i)
    {
        if(i > succ.size()) return nullptr;
        else return succ[i];
    }

    void Node::dump_func()
    {

    }

    void Node::dump_inst()
    {

    }

    void Node::add_input_value(Value * v)
    {

    }

    void Node::add_output_value(Value * v)
    {

    }

    std::vector<Value*> Node::get_input_value()
    {

    }

    std::vector<Value*> Node::get_output_value()
    {

    }

    InstNode::InstNode(Instruction * target_inst)
    {
        inst = target_inst;
    }

    FuncNode::FuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f)
    {
        call_inst = ci;
        called_func = cf;
        gpu_flag = gpu_f;
        has_graph = graph_f;
    }

    std::string FuncNode::get_func_name()
    {
        return called_func->getName().str();
    }

    void FuncNode::dump_func()
    {
        std::cout<<get_func_name()<<std::endl;
    }

    void FuncNode::dump_inst()
    {
        errs()<<*call_inst<<"\n";
    }

    GPUFuncNode::GPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    CPUFuncNode::CPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    MemcpyNode::MemcpyNode(CallInst * ci, Function * cf, bool gpu_f = false, bool graph_f = false):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    Seq_Graph::Seq_Graph():Node()
    {
        Entry_Node = new Node();
        End_Node = new Node();
        Entry_Node->clear_pred();
        Entry_Node->add_succ(End_Node);
        End_Node->add_pred(Entry_Node);
        End_Node->clear_succ();
    }

    void Seq_Graph::Insert(Node * target_node, Node * pred_node)
    {
        Node * next_node = pred_node->get_succ(0);
        pred_node->clear_succ();
        pred_node->add_succ(target_node);
        target_node->add_pred(pred_node);
        target_node->add_succ(next_node);
        next_node->clear_pred();
        next_node->add_pred(target_node);
    }

    Node * Seq_Graph::get_last_Node()
    {
        Node * last_node = End_Node->get_pred(0);
        return last_node;
    }

    bool Seq_Graph::IsEndNode(Node * node)
    {
        return node == End_Node;
    }

    Node * Seq_Graph::getEntryNode()
    {
        return Entry_Node;
    }

    void Seq_Graph::WalkGraph()
    {
        //TO.DO.
    }

    void Seq_Graph::Print_allNode()
    {
        Node * cur_node = getEntryNode();
        cur_node = cur_node->get_succ(0);
        while(!IsEndNode(cur_node))
        {
            if(cur_node == nullptr)
            {
                std::cout<<"Cannot get func_node of cur_node"<<std::endl;
                exit(1);
            }
            cur_node->dump_func();
            //cur_node->dump_inst();
            cur_node = cur_node->get_succ(0);
        }
        return;
    }
    
}