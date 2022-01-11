#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<queue>
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
#include "llvm/IR/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"


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

    size_t Node::get_pred_num()
    {
        return pred.size();
    }

    size_t Node::get_succ_num()
    {
        return succ.size();
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
        input_value.push_back(v);
    }

    void Node::add_output_value(Value * v)
    {
        output_value.push_back(v);
    }

    std::vector<Value*> Node::get_input_value()
    {
        return input_value;
    }

    std::vector<Value*> Node::get_output_value()
    {
        return output_value;
    }

    CallInst * Node::getCallInst()
    {
        return nullptr;
    }

    Function * Node::getFunction()
    {
        return nullptr;
    }

    void Node::addBB(BasicBlock * bb)
    {
        bbs.push_back(bb);
    }

    void Node::deleteBB(BasicBlock * bb)
    {
        size_t i = 0;
        for(; i < bbs.size(); i++)
        {
            if(bb == bbs[i])
                break;
            else continue;
        }
        if(i == bbs.size())
        {
            std::cout<<"Cannot delete bb\n";
        }
        else 
        {
            bbs.erase(bbs.begin() + i);
        }
    }

    void Node::dumpBBs()
    {
        for(BasicBlock * bb : bbs)
        {
            bb->printAsOperand(errs(),false);
            errs()<<" ";
        }
        errs()<<"\n";
    }

    void Node::CollectBBs(Module & M)
    {
        //do nothing
    }

    std::vector<BasicBlock*> Node::getBBs()
    {
        std::vector<BasicBlock*> res;
        size_t num = bbs.size();
        for(size_t i = 0; i < num; i++)
        {
            res.push_back(bbs[i]);
        }
        return res;
    }

    InstNode::InstNode(Instruction * target_inst)
    {
        inst = target_inst;
    }

    InstNode::~InstNode()
    {
        //do nothing.
    }

    FuncNode::FuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f)
    {
        call_inst = ci;
        called_func = cf;
        gpu_flag = gpu_f;
        has_graph = graph_f;
    }

    FuncNode::~FuncNode()
    {
        //nothing to do
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

    CallInst * FuncNode::getCallInst()
    {
        return call_inst;
    }

    Function * FuncNode::getFunction()
    {
        return called_func;
    }

    void FuncNode::CollectBBs(Module & M)
    {
        //do nothing
    }

    void FuncNode::SetStream(GlobalValue * stream)
    {
        //do nothing
    }

    GPUFuncNode::GPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    void GPUFuncNode::CollectBBs(Module & M)
    {
        BasicBlock * cur_bb = this->getCallInst()->getParent();
        //Normally, in sequential kernel launches, one bb containing a kernel only has one predecessor
        //QUES.: How about if-else or loop?
        //dump_inst();
        //errs()<<"Its bb has "<<pred_size(cur_bb)<<" pred bbs\n";
        std::vector<BasicBlock*> bbs_to_split;
        std::vector<Instruction*> splited_node_insts;
        for(BasicBlock * direct_pred_bb : predecessors(cur_bb))
        {
            for(auto inst = direct_pred_bb->begin(), inst_end = direct_pred_bb->end();
                inst != inst_end; inst++)
            {
                if(isa<CallInst>(inst))
                {
                    CallInst * PushCallConfig_Inst = dyn_cast<CallInst>(inst);
                    if(PushCallConfig_Inst->getCalledFunction()->getName().str() == "__hipPushCallConfiguration")
                    {
                        //errs()<<"CHICHI\n";
                        //Instruction * stream_def_inst = dyn_cast<Instruction>(stream_v);
                        //QUES.: In loop, it will skip the phi node, but we dont want to carry the 
                        if(PushCallConfig_Inst == dyn_cast<Instruction>(direct_pred_bb->getFirstInsertionPt()))
                        {
                            addBB(direct_pred_bb);
                            continue;
                        }
                        else
                        {
                            //partitioned the direct_pred_bb;
                            //BasicBlock * new_pred_bb = SplitBlock(direct_pred_bb,PushCallConfig_Inst);
                            //addBB(new_pred_bb);
                            bbs_to_split.push_back(direct_pred_bb);
                            splited_node_insts.push_back(PushCallConfig_Inst);
                        }
                    }
                }
            }
        }

        for(size_t i = 0; i < bbs_to_split.size(); i++)
        {
            BasicBlock * new_pred_bb = SplitBlock(bbs_to_split[i],splited_node_insts[i]);
            addBB(new_pred_bb);
        }
        addBB(cur_bb);
        //errs()<<"Finish CHICHI\n";
    }

    void GPUFuncNode::SetStream(GlobalValue * stream_v)
    {
        std::vector<BasicBlock*> own_bbs = this->getBBs();
        Instruction * PushCallConfig_Inst = nullptr;

        //Locate the hipPushCallConfig inst, which is the first inst of its BB
        for(BasicBlock * bb : own_bbs)
        {
            for(auto inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
            {
                if(isa<CallInst>(inst))
                {
                    CallInst * call_inst = dyn_cast<CallInst>(inst);
                    Function * called_func = call_inst->getCalledFunction();
                    if(called_func == nullptr) continue;
                    else
                    {
                        if(called_func->getName().str() == "__hipPushCallConfiguration")
                        {
                            PushCallConfig_Inst = call_inst;
                        }
                    }
                }
            }
        }

        //insert the load inst of stream
        IRBuilder<> builder(PushCallConfig_Inst);
        Value * cur_stream = builder.CreateLoad(stream_v,"");
        if(cur_stream == nullptr)
        {
            errs()<<"Cannot create stream load inst\n";
            exit(1);
        }
        
        //replace the stream arg in pushCallConfig_Inst
        PushCallConfig_Inst->setOperand(5,cur_stream);
    }

    CPUFuncNode::CPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    MemcpyNode::MemcpyNode(CallInst * ci, Function * cf, bool gpu_f = false, bool graph_f = false):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //Init the input/output of memcpy node
        Value * input_arg = ci->getArgOperand(1);
        Value * output_arg = ci->getArgOperand(0);

        //TO.DO.: Need to identify whether we need to bitcast pointer type to i8 *. If passed a i8*, then bitcast is unneccessary
        Instruction * input_bitcast_inst = dyn_cast<Instruction>(input_arg);
        Instruction * output_bitcast_inst = dyn_cast<Instruction>(output_arg);

        Value * input_v = input_bitcast_inst->getOperand(0);
        Value * output_v = output_bitcast_inst->getOperand(0);

        add_input_value(input_v);
        add_output_value(output_v);
    }

    void MemcpyNode::CollectBBs(Module & M)
    {
        //errs()<<"KAKA\n";
        CallInst * callmemcpy_inst = this->getCallInst();
        BasicBlock * cur_bb = callmemcpy_inst->getParent();
        Instruction * partitioned_node = callmemcpy_inst->getNextNode();
        //NOTE: Or we can use getNextNonDebugInstruction() to skip debug instructions
        SplitBlock(cur_bb,partitioned_node);
        this->addBB(cur_bb);
        //errs()<<"Finish KAKA\n";
    }

    void MemcpyNode::SetStream(GlobalValue * stream)
    {
        //need to replace it with memcpyAsync()
        
        //then give it a stream
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

    Seq_Graph::~Seq_Graph()
    {
        delete Entry_Node;
        delete End_Node;
        Entry_Node = nullptr;
        End_Node = nullptr;
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
            //cur_node->dump_func();
            cur_node->dump_inst();
            std::vector<Value*> input_v = cur_node->get_input_value();
            std::vector<Value*> output_v = cur_node->get_output_value();
            errs()<<"Input Value:\n";
            for(auto v : input_v)
            {
                errs()<<*v<<"\n";
            }
            errs()<<"Output Value:\n";
            for(auto v : output_v)
            {
                errs()<<*v<<"\n";
            }
            
            //cur_node->dump_inst();
            cur_node = cur_node->get_succ(0);
        }
        return;
    }
    
    DAG::DAG():Node()
    {
        Entry_Node = new Node();
        End_Node = new Node();
        Entry_Node->clear_pred();
        End_Node->clear_succ();
        Entry_Node->add_succ(End_Node);
        End_Node->add_pred(Entry_Node);
    }

    DAG::~DAG()
    {
        delete Entry_Node;
        Entry_Node = nullptr;
        delete End_Node;
        End_Node = nullptr;
    }

    void DAG::ConstructFromSeq_Graph(Seq_Graph * SG)
    {
        std::vector<Node*> empty_vec;
        pred_map[Entry_Node] = empty_vec;
        pred_map[End_Node].push_back(Entry_Node);

        Node * Seq_EntryNode = SG->getEntryNode();
        Node * Seq_G_cur_node = Seq_EntryNode->get_succ(0);
        while(!SG->IsEndNode(Seq_G_cur_node))
        {
            this->Insert(Seq_G_cur_node);
            Seq_G_cur_node = Seq_G_cur_node->get_succ(0);
        }
    }

    void DAG::Insert(Node * node)                   //Here we get the func_node in Seq_Graph
    {
        //Get all needed info of node to init node in dag
        std::vector<Value *> input_v = node->get_input_value();
        std::vector<Value *> output_v = node->get_output_value();
        std::vector<BasicBlock *> bbs = node->getBBs();
        //errs()<<"Inserting ";
        //node->dump_inst();

        CallInst * call_inst = node->getCallInst();
        Function * called_func = node->getFunction();
        
        
        //QUES.: Should we specify it as categoried function?
        FuncNode * new_node = new FuncNode(call_inst,called_func,true,true);
        for(BasicBlock * bb : bbs) new_node->addBB(bb);
        
        //Here we should follow the WAW, WAR, RAW order, so we should find the last pred node using following rule:
        //1. compare new_node's input value to all possible pred's output
        //2. compare new_node's output value to all possible pred's input
        //3. compare new_node's output value to all possible pred's output
        std::vector<Node *> pred_nodes;

        //RAW
        for(size_t i = 0; i < input_v.size(); i++)
        {
            //Here we make all read and write value of current kernel function into its input_v
            Value * cur_v = input_v[i];
            Type * type = cur_v->getType();
            if(type->isPointerTy())
            {
                new_node->add_input_value(cur_v);
                
                Node * pred_node = reverse_find_pred(cur_v,true,true)[0];
                
                /*
                if(pred_node == Entry_Node) errs()<<"raw_Pre_Node is Entry Node\n";
                else pred_node->dump_inst();
                */
                
                if(pred_node==nullptr)
                {
                    errs()<<"Cannot find the raw_pred_node\n";
                    exit(1);
                }
                //NOTE: Since we only return one node, so we should avoid multiple nodes outputing the same value
                //in the same level. TO.DO.: When inserting node, we should follow WAW order.
                //We should check whether pred_node is right before End_Node, 
                //if so, the new_node will be inserted between them, End_Node should be cleared from succ of pred_node
                //if not, End_Node should not be cleared
                /*                
                bool Last_Node_Flag = false;
                size_t pred_node_succ_num = pred_node->get_succ_num();
                //Determine whether it is the last tail of current branch like entry->xxx->(new_node)->end_node
                for(size_t j = 0; j < pred_node_succ_num; j++)
                {
                    Node * pred_node_succ = pred_node->get_succ(j);
                    if(pred_node_succ == End_Node)
                    {
                        Last_Node_Flag = true;
                        break;
                    }
                }
                */
                //if(Last_Node_Flag)  pred_node->clear_succ(End_Node);
                //pred_node->add_succ(new_node);
                pred_nodes.push_back(pred_node);
            }
            else
                continue;
        }

        //WAW & WAR
        for(size_t i = 0; i < output_v.size(); i++)
        {
            Value * cur_v = output_v[i];
            Type * type = cur_v->getType();
            if(type->isPointerTy())
            {
                new_node->add_output_value(cur_v);
                
                Node * waw_pred_node = reverse_find_pred(cur_v,true,true)[0];
                if(waw_pred_node==nullptr)
                {
                    errs()<<"Cannot find the waw_pred_node\n";
                    exit(1);
                }
                /*
                if(waw_pred_node == Entry_Node) errs()<<"waw_Pre_Node is Entry Node\n";
                else waw_pred_node->dump_inst();
                */
                std::vector<Node *> war_pred_nodes = reverse_find_pred(cur_v,false,false);

                /*
                errs()<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
                node->dump_inst();
                errs()<<"WAR Pred Nodes:\n";
                for(auto war_pred_node : war_pred_nodes) war_pred_node->dump_inst();
                errs()<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
                */

                if(war_pred_nodes.size()==0)
                {
                    errs()<<"Cannot find the war_pred_node\n";
                    exit(1);
                }
                /*
                if(war_pred_node == Entry_Node) errs()<<"war_Pre_Node is Entry Node\n";
                else war_pred_node->dump_inst();
                */
                /*
                bool waw_pred_node_Last_Node_Flag = false;
                bool war_pred_node_Last_Node_Flag = false;
                size_t waw_pred_node_succ_num = waw_pred_node->get_succ_num();
                size_t war_pred_node_succ_num = war_pred_node->get_succ_num();
                for(size_t j = 0; j < waw_pred_node_succ_num; j++)
                {
                    if(waw_pred_node->get_succ(j) == End_Node)
                    {
                        waw_pred_node_Last_Node_Flag = true;
                        break;
                    }
                }
                for(size_t j = 0; j < war_pred_node_succ_num; j++)
                {
                    if(war_pred_node->get_succ(j) == End_Node)
                    {
                        war_pred_node_Last_Node_Flag = true;
                        break;
                    }
                }
                */
                //if(war_pred_node_Last_Node_Flag) war_pred_node->clear_succ(End_Node);
                //if(waw_pred_node_Last_Node_Flag) waw_pred_node->clear_succ(End_Node);

                //war_pred_node->add_succ(new_node);
                //waw_pred_node->add_succ(new_node);
                for(auto war_pred_node : war_pred_nodes) pred_nodes.push_back(war_pred_node);
                pred_nodes.push_back(waw_pred_node);
            }
            else continue;
        }

        new_node->add_succ(End_Node);
        End_Node->add_pred(new_node);
        pred_map[End_Node].push_back(new_node);

        //Delete EntryNode from pred of End_Node
        /*
        bool entry_node_is_pred_of_end_node = false;
        size_t EndNode_pred_num = End_Node->get_pred_num();
        for(size_t i = 0; i < EndNode_pred_num; i++)
        {
            if(End_Node->get_pred(i) == Entry_Node)
            {
                entry_node_is_pred_of_end_node = true;
                break;
            }
        }
        if(entry_node_is_pred_of_end_node) End_Node->clear_pred(Entry_Node);
        */
        //Make them unique and remove EntryNode from them if they consist more than EntryNode
        std::unordered_map<Node*, bool> unique_pred_node_m;
        for(Node * pred_node : pred_nodes)
        {
            if(unique_pred_node_m.find(pred_node) != unique_pred_node_m.end()) continue;
            else unique_pred_node_m[pred_node] = true;
        }

        if(unique_pred_node_m.size() > 1 && unique_pred_node_m.find(Entry_Node) != unique_pred_node_m.end())
        {
            unique_pred_node_m.erase(Entry_Node);
        }

        for(auto it = unique_pred_node_m.begin(); it != unique_pred_node_m.end(); it++)
        {
            Node * pred_node = it->first;
            //clear EndNode out of pred_node's succ if they are adjunt before
            bool clear_end_node = false;
            size_t succ_num = pred_node->get_succ_num();
            for(size_t i = 0; i < succ_num; i++)
            {
                if(pred_node->get_succ(i) == End_Node)
                    clear_end_node = true;
            }
            if(clear_end_node) 
            {
                pred_node->clear_succ(End_Node);
                End_Node->clear_pred(pred_node);
                //delete pred_node from End_Node in pred_map
                size_t index = 0;
                for(Node * pred : pred_map[End_Node])
                {
                    if(pred == pred_node)
                        break;
                    else index++;
                }
                if(index < pred_map[End_Node].size())
                    pred_map[End_Node].erase(pred_map[End_Node].begin() + index);

            }

            pred_node->add_succ(new_node);
            new_node->add_pred(pred_node);
            pred_map[new_node].push_back(pred_node);
        }
        /*
        std::unordered_map<Node*,bool> m;
        for(Node * pred_node : pred_nodes)
        {
            if(m.find(pred_node) == m.end()) 
            {
                new_node->add_pred(pred_node);
                pred_map[new_node].push_back(pred_node);
                //clear EndNode out of pred_node's succ if they are adjunt before
                bool clear_end_node_flag = false;
                size_t succ_num = pred_node->get_succ_num();
                for(size_t i = 0; i < succ_num; i++)
                {
                    if(pred_node->get_succ(i) == End_Node)
                        clear_end_node_flag = true;
                }
                if(clear_end_node_flag) pred_node->clear_succ(End_Node);
                
                pred_node->add_succ(new_node);
                m[pred_node] = true;
            }
            else 
                continue;
        }
        */
    }

    void DAG::Delete(Node * node)
    {
        //TO.DO.
    }

    std::vector<Node*> DAG::reverse_find_pred(Value * v, bool find_pred_output, bool Only_flag)             //Only_flag is true means we are dealing WAW or RAW
                                                                                                            //because one value only can be modified once in one layer
                                                                                                            //and that's the latest pred
    {
        std::queue<Node*> q;
        q.push(End_Node);
        std::vector<Node*> ans;
        while(!q.empty())
        {
            int n = q.size();
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                q.pop();
                if(cur_node == Entry_Node)
                {
                    ans.push_back(Entry_Node);
                }
                std::vector<Value*> target_values = find_pred_output ? cur_node->get_output_value() : cur_node->get_input_value();
                for(auto target_v : target_values)
                {
                    if(target_v == v)
                    {
                        ans.push_back(cur_node);
                        if(Only_flag) return ans;
                        else continue;
                    }
                }
                if(cur_node == Entry_Node) continue;
                size_t cur_pred_num = cur_node->get_pred_num();
                for(size_t j = 0; j < cur_pred_num; j++) 
                {
                    q.push(cur_node->get_pred(j));
                }
            }
        }
        
        return ans;
    }

    void DAG::dump()
    {
        //pred_map contains preds of nodes except Entry Node and End Node
        //Need to fullfill it
        
        
        errs()<<"Dumping pred_map\n";
        for(auto it = pred_map.begin(); it != pred_map.end(); it++)
        {
            errs()<<"Target node: ";
            if(it->first == Entry_Node) errs()<<"Entry Node\n";
            else if(it->first == End_Node) errs()<<"End Node\n";
            else it->first->dump_inst();

            for(auto pred : it->second)
            {
                errs()<<"Depends on ";
                if(pred == Entry_Node) errs()<<"Entry_Node\n";
                else pred->dump_inst();
            }
            errs()<<"\n";
        }

        std::unordered_map<Node *, std::vector<Node*>> rudu_map = pred_map;
        std::unordered_map<Node *, bool> vis_m;
        
        std::queue<Node*> q;
        q.push(Entry_Node);
        vis_m[Entry_Node] = true;
        //Only node with rudu = 0 can be in Queue
        int level = 0;
        while(!q.empty())
        {
            std::cout<<"********************\n";
            std::cout<<"Level "<<level++<<" :"<<std::endl;
            int n = q.size();
            if(n==1 && q.front() == End_Node)
            {
                errs()<<"End_Node\n";
                std::cout<<"********************\n";
                break;
            }
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                if(cur_node != Entry_Node)
                    cur_node->dump_inst();
                else
                    errs()<<"Entry Node\n";
                
                q.pop();
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    int target_index = 0;
                    int n_pred = it->second.size();
                    for(Node * pred : it->second)
                    {
                        if(pred == cur_node) break;
                        else target_index++;
                    }
                    if(target_index >= n_pred) continue;
                    else it->second.erase(it->second.begin() + target_index);
                }
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    if(!vis_m[it->first] && it->second.size() == 0) 
                    {
                        q.push(it->first);
                        vis_m[it->first] = true;
                    }
                }
            }
            std::cout<<"********************\n";
        }
    }

    void DAG::levelize()
    {
        std::unordered_map<Node *, std::vector<Node*>> rudu_map = pred_map;
        std::unordered_map<Node *, bool> vis_m;
        
        std::queue<Node*> q;
        q.push(Entry_Node);
        vis_m[Entry_Node] = true;
        //Only node with rudu = 0 can be in Queue
        int level = 0;

        while(!q.empty())
        {
            //std::cout<<"********************\n";
            //std::cout<<"Level "<<level++<<" :"<<std::endl;
            int n = q.size();
            if(n==1 && q.front() == End_Node)
            {
                node_level_map[End_Node] = level;
                level_nodes_map[level].push_back(End_Node);
                break;
            }
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                node_level_map[cur_node] = level;
                level_nodes_map[level].push_back(cur_node);
                q.pop();
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    int target_index = 0;
                    int n_pred = it->second.size();
                    for(Node * pred : it->second)
                    {
                        if(pred == cur_node) break;
                        else target_index++;
                    }
                    if(target_index >= n_pred) continue;
                    else it->second.erase(it->second.begin() + target_index);
                }
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    if(!vis_m[it->first] && it->second.size() == 0) 
                    {
                        q.push(it->first);
                        vis_m[it->first] = true;
                    }
                }
            }
            level++;
        }
        n_level = level-1;
    }

    void DAG::dump_level()
    {
        errs()<<"Dump Level Map\n";
        errs()<<"------------------\n";
        for(auto it = level_nodes_map.begin(); it != level_nodes_map.end(); it++)
        {
            errs()<<"Level "<<it->first<<" :\n";
            if(it->second.size() == 1 && it->second[0] == Entry_Node) errs()<<"Entry Node\n";
            else if(it->second.size() == 1 && it->second[0] == End_Node) errs()<<"End Node\n";
            else 
            {
                for(auto node : it->second)
                {
                    node->dump_inst();
                }
            }
        }
        errs()<<"----------------\n";
    }

    //Function to help sortPredByUndoneSucc
    bool cmp(std::pair<Node*,size_t> a, std::pair<Node*,size_t> b)
    {
        return a.second < b.second;
    }

    void DAG::StreamDistribute(StreamGraph * StreamG)
    {
        if(!n_level)                                                    //n_level means level num except Entry/End
        {
            errs()<<"Wrong Levelize DAG\n";
            exit(1);
        }
        
        size_t stream_num = StreamG->get_stream_num();
        
        for(size_t i = 1; i <= n_level; i++)
        {
            std::vector<Node*> nodes = level_nodes_map[i];
            std::unordered_map<Node*,bool> setted_flag_map;
            std::unordered_map<Node*,size_t> node_original_index_map;
            for(auto node: nodes) setted_flag_map[node] = false;
            
            //hash for those have no preds except EntryNode
            for(size_t node_index = 0; node_index < nodes.size(); node_index++)
            {
                Node * cur_node = nodes[node_index];
                node_original_index_map[cur_node] = node_index;
                if(cur_node->get_pred_num()==1 && cur_node->get_pred(0) == Entry_Node)
                {
                    size_t stream_id = node_index % stream_num;
                    StreamG->node_set_stream(cur_node,stream_id);
                    StreamG->init_node_undone_succ(cur_node);
                    //Entry_Node doesn't exist in StreamGraph, no need to reduce its undone succ
                    //Also, there is no EventEdge produced from this
                    setted_flag_map[cur_node] = true;
                }
            }

            //For the first level, work are done
            if(i == 1) continue;
            //Sort Nodes by their pred(from small to big)
            SortByPredNum(nodes,0,nodes.size()-1);

            for(auto node : nodes)
            {
                if(setted_flag_map[node]) continue;

                int stream_end_pred = 0;             //when it's true, means that we need to find the optimal
                                                            //pred in those which are the end of streams
                std::vector<Node*> stream_end_preds;

                size_t pred_num = node->get_pred_num();
                std::vector<std::pair<Node*,size_t>> preds;
                //Sort pred by their undone succ(increase order)
                //Collect preds which are end of their streams
                for(size_t j = 0; j < pred_num; j++)
                {
                    Node * cur_pred = node->get_pred(j);
                    if(StreamG->node_is_end_of_stream(cur_pred)) 
                    {
                        stream_end_pred++;
                        stream_end_preds.push_back(cur_pred);
                    }
                    preds.push_back(std::pair<Node*,size_t>(cur_pred,StreamG->get_node_undone_succ(cur_pred)));
                }
                std::sort(preds.begin(), preds.end(), cmp);
                
                //at least one node will be undone succ for each pred(cur node)
                //It means no zero in preds' values
                
                if(!stream_end_pred)                //Situation 1. No optimal stream can be selected
                {
                    //random pick one
                    size_t stream_id = node_original_index_map[node] % stream_num;
                    StreamG->node_set_stream(node,stream_id);
                    StreamG->init_node_undone_succ(node);
                    //Add EventEdges
                    for(size_t j = 0; j < pred_num; j++)
                    {
                        Node * cur_pred = node->get_pred(j);
                        StreamG->reduce_node_undone_succ(cur_pred);                 //QUES.: Should the undone_succ_num only affect on the pred which is end of stream?
                        if(StreamG->get_node_stream(cur_pred) != stream_id)
                        {
                            StreamG->add_EE(cur_pred,node);
                        }
                    }
                    setted_flag_map[node] = true;
                }
                else if(stream_end_pred == 1)       //Situation 2. only have one pred which is end of its stream
                {
                    
                    Node * target_pred = stream_end_preds[0];
                    size_t stream_id = StreamG->get_node_stream(target_pred);
                    StreamG->node_set_stream(node,stream_id);
                    StreamG->init_node_undone_succ(node);
                    //Add EventEdges
                    for(size_t j = 0; j < pred_num; j++)
                    {
                        Node * cur_pred = node->get_pred(j);
                        StreamG->reduce_node_undone_succ(cur_pred);                 //QUES.: Should the undone_succ_num only affect on the pred which is end of stream?
                        if(StreamG->get_node_stream(cur_pred) != stream_id)
                        {
                            StreamG->add_EE(cur_pred,node);
                        }
                    }
                    setted_flag_map[node] = true;
                }
                else                                //Situation 3. multiple quanlified preds which are end of their stream s
                {
                    //selece the pred with least undone succ
                    for(auto it = preds.begin(), it_end = preds.end(); it != it_end; it++)
                    {
                        Node * target_pred = it->first;
                        std::vector<Node*>::iterator finder = std::find(stream_end_preds.begin(), stream_end_preds.end(), target_pred);
                        if(finder == stream_end_preds.end()) continue;
                        else
                        {
                            size_t stream_id = StreamG->get_node_stream(target_pred);
                            StreamG->node_set_stream(node,stream_id);
                            StreamG->init_node_undone_succ(node);
                            for(size_t j = 0; j < pred_num; j++)
                            {
                                Node * cur_pred = node->get_pred(j);
                                StreamG->reduce_node_undone_succ(cur_pred);
                                if(StreamG->get_node_stream(cur_pred) != stream_id) StreamG->add_EE(cur_pred,node);
                            }
                            setted_flag_map[node] = true;
                            break;
                        }
                    }
                }

            }
        }
    }

    int find_pivot(std::vector<Node*> & p, int s, int e)
    {
        size_t pivot = p[s]->get_pred_num();
        int bigger_index = e+1;
        for(int i = e; i > s; i--)
        {
            if(p[i]->get_pred_num() >= pivot)
            {
                bigger_index--;
                Node * tmp = p[i];
                p[i] = p[bigger_index];
                p[bigger_index] = tmp;
            }
        }
        bigger_index--;
        Node * tmp = p[s];
        p[s] = p[bigger_index];
        p[bigger_index] = tmp;
        return bigger_index;
    }

    void DAG::SortByPredNum(std::vector<Node*> & p, int s, int e)
    {
        if(s>=e) return;

        int pivot_index = find_pivot(p,s,e);
        SortByPredNum(p,s,pivot_index-1);
        SortByPredNum(p,pivot_index+1,e);
    }

    EventEdge::EventEdge(Node * a, Node * b)
    {
        prev = a;
        succ = b;
    }

    void EventEdge::dump()
    {
        errs()<<"Prev: ";
        prev->dump_inst();
        errs()<<"Succ: ";
        succ->dump_inst();
    }

    StreamGraph::StreamGraph()
    {
        //do nothing
    }

    StreamGraph::StreamGraph(size_t n)
    {
        stream_n = n;
        for(size_t i = 0; i < n; i++)
        {
            stream_nodes_map[i] = std::vector<Node*>();
        }
    }

    StreamGraph::~StreamGraph()
    {
        //do nothing
    }

    void StreamGraph::add_EE(Node * pred, Node * cur)
    {
        size_t pred_stream_id = get_node_stream(pred);
        size_t pred_stream_index = get_node_stream_index(pred,pred_stream_id);
        std::vector<Node*> same_stream_preds;
        size_t pred_num = cur->get_pred_num();
        for(size_t i = 0; i < pred_num; i++)
        {
            Node * cur_pred = cur->get_pred(i);
            size_t cur_pred_stream_id = get_node_stream(cur_pred);
            size_t cur_pred_stream_index = get_node_stream_index(cur_pred,cur_pred_stream_id);
            if(cur_pred_stream_id != pred_stream_id) continue;
            else
            {
                if(pred_stream_index > cur_pred_stream_index)
                {
                    delete_EE(cur_pred,cur);
                }
            }
        }
        EEs.push_back(EventEdge(pred,cur));
    }

    void StreamGraph::delete_EE(Node * pred, Node * cur)
    {
        //delete the EE
        size_t target_index = 0;
        for(size_t i = 0; i < EEs.size(); i++)
        {
            if(EEs[i].prev == pred && EEs[i].succ == cur)
            {
                target_index = i;
            }
        }
        if(target_index == EEs.size())
        {
            errs()<<"Cannot delete EE\n";
            exit(1);
        }
        else
        {
            EEs.erase(EEs.begin() + target_index);
        }
    }

    size_t StreamGraph::get_EE_num()
    {
        return EEs.size();
    }

    void StreamGraph::node_set_stream(Node * cur, size_t stream_id)
    {
        node_stream_map[cur] = stream_id;
        stream_nodes_map[stream_id].push_back(cur);
    }

    size_t StreamGraph::get_node_stream(Node * cur)
    {
        if(node_stream_map.find(cur) == node_stream_map.end())
        {
            errs()<<"Not recorded in node_stream_map\n";
            exit(1);
        }
        else
        {
            return node_stream_map[cur];
        }
    }

    size_t StreamGraph::get_node_stream_index(Node * node, size_t stream_id)
    {
        size_t index = 0;
        for(Node * target_node : stream_nodes_map[stream_id])
        {
            if(target_node == node) break;
            else index++;
        }
        if(index == stream_nodes_map[stream_id].size()) 
        {
            errs()<<"Cannot get the index of node\n";
            exit(1);
        }
        return index;
    }

    bool StreamGraph::node_is_end_of_stream(Node * cur)
    {
        for(auto it = stream_nodes_map.begin(), it_end = stream_nodes_map.end(); it != it_end; it++)
        {
            if(it->second.back() == cur) return true;
        }
        return false;
    }

    void StreamGraph::init_node_undone_succ(Node * node)
    {
        //We count EndNode in DAG here but it wont be the node we need to operate in StreamGraph
        size_t succ_num = node->get_succ_num();
        unset_succ_num_map[node] = succ_num;
    }

    void StreamGraph::reduce_node_undone_succ(Node * cur)
    {
        if(unset_succ_num_map.find(cur) == unset_succ_num_map.end())
        {
            errs()<<"Not recorded in unset_succ_num_map\n";
            exit(1);
        }
        else
        {
            unset_succ_num_map[cur]--;
        }
    }

    size_t StreamGraph::get_node_undone_succ(Node * cur)
    {
        if(unset_succ_num_map.find(cur) == unset_succ_num_map.end())
        {
            errs()<<"Not recorded in unset_succ_num_map\n";
            exit(1);
        }
        else
        {
            return unset_succ_num_map[cur];
        }
    }

    size_t StreamGraph::get_stream_num()
    {
        return stream_n;
    }    

    void StreamGraph::dump_Graph()
    {
        errs()<<"**********************\n";
        errs()<<"StreamGraph:\n";
        for(auto it = stream_nodes_map.begin(), it_end = stream_nodes_map.end(); it != it_end; it++)
        {
            size_t stream_id = it->first;
            errs()<<"The "<<stream_id<<"th Stream:\n";
            for(Node * node : it->second)
            {
                node->dump_inst();
            }
        }
        errs()<<"**********************\n";
    }

    void StreamGraph::dump_EEs()
    {
        errs()<<"**********************\n";
        errs()<<"EventEdge:\n";
        for(size_t i = 0; i < EEs.size(); i++)
        {
            EEs[i].dump();
        }
        errs()<<"**********************\n";

    }

}