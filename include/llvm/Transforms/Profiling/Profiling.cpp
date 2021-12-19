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
#include "llvm/Analysis/LoopInfo.h"
#include<unordered_map>
#include "Graph.h"

using namespace llvm;
using namespace graph;
//This pass is to insert timing function before & after each function and also analysis 
//data dependencies among functions.

//Construct graph for each function

namespace{

    struct Profiling : public ModulePass{
        static char ID;

        Profiling();
        ~Profiling();

        bool runOnModule(Module & M) override;
        std::string get_father_dir(std::string);
        void get_tool_library_func(Module & M);
        void getAnalysisUsage(AnalysisUsage &AU) const override;
        std::vector<Value*> GetKernelArgs(Node *, Module & M);
        void set_target_func(Module & M);                               //To locate target functions including multiple kernel func

        //data
        std::string profiling_data_file_path;
        std::unordered_map<Function*,bool> tool_func_map;
        std::unordered_map<CallInst*,Node*> callinst_node_map;
        std::unordered_map<Function*,bool> target_func_map;
        std::vector<std::string> target_func_list;                      //We only have unmangling function name
    };

    Profiling::Profiling() : ModulePass(ID){
        target_func_list= {"FUNC"};
    }

    Profiling::~Profiling(){

    }

    void Profiling::getAnalysisUsage(AnalysisUsage &AU) const{
        AU.addRequired<LoopInfoWrapperPass>();
    }

    bool Profiling::runOnModule(Module & M){
        std::string father_path;
        std::string ModuleName = M.getModuleIdentifier();
        father_path = get_father_dir(ModuleName);
        std::string profiling_data_file_path = father_path+"profiling.txt";
        std::string data_dependency_graph_file_path = father_path+"DDG.txt";

        //This is to profile
        //TO.DO.: need an extra function to output the timing into file(TOOL LIBRARY)
        /*
        auto hipEvent_t = StructType::create(M.getContext(), "struct.ihipEvent_t");
        auto hipEvent_PtrT = PointerType::get(hipEvent_t,0);
        GlobalVariable * gpu_start_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_start");
        GlobalVariable * gpu_end_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_end");

        gpu_start_event_ptr->setAlignment(MaybeAlign(8));
        gpu_end_event_ptr->setAlignment(MaybeAlign(8));
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(hipEvent_PtrT);
        if(NULLPTR == nullptr)
        {
            errs()<<"cannot gen nullptr for hipevent\n";
            exit(1);
        }
        gpu_start_event_ptr->setInitializer(NULLPTR);
        gpu_end_event_ptr->setInitializer(NULLPTR);
        */

        //TO.DO.: Create global stream
        //1. define stream_type
        //2. announce global stream object
        //3. use hipcreateStream() to init stream
        auto hipStream_t = StructType::getTypeByName(M.getContext(), "struct.ihipStream_t");
        auto hipStream_t_PtrT = PointerType::get(hipStream_t,0);
        auto hipStream_t_PtrPtrT = PointerType::get(hipStream_t_PtrT,0);
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(hipStream_t_PtrT);

        GlobalVariable * stream_var_ptrs [10];
        for(int i = 0; i < 10; i++)
        {
            stream_var_ptrs[i] = new GlobalVariable(M,hipStream_t_PtrT,false,GlobalValue::CommonLinkage,0,"mzw_s"+std::to_string(i));
            stream_var_ptrs[i]->setAlignment(MaybeAlign(8));
            
            stream_var_ptrs[i]->setInitializer(NULLPTR);
        }

        //TO.DO.: use hipcreateStream() to init stream        

        //In order to construct the ddg of each function(CPU function, GPU kernel and Memcpy)
        //First we need to identify them
        //For Kernel, we will use hipLaunchKernel to call kernel, so use it to recognize
        //For Memcpy, we will use hipMemcpy, so use it to recognize
        //For regular CPU function, we need user to add prefix of "cpu_" of each CPU function
        //Other functions in HIP and library functions will be treated as CPU functions.

        get_tool_library_func(M);
        set_target_func(M);

        //In fact, we need to handle all branch and loop first, just make them as a sub-graph
        //then hash them into a set, in the second walk(handle the sequential instruction) ingnore
        //those hashed inst.
        //Also, in the first walk, should we identify input and output of each function?
        //TO.DO.:  Handle the Loop Graph
        /*
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(tool_func_map.find(cur_func) != tool_func_map.end()) continue;
            else if(cur_func->size() == 0) continue;

            LoopInfo & LI = getAnalysis<LoopInfoWrapperPass>(*cur_func).getLoopInfo();
            size_t loop_counter = 0;
            for(Loop * L : LI)        //this is for every outmost(top-level) loop?
            {
                errs()<<"Loop "<<++loop_counter<<"\n";
                //11-19 consider one bb, not nested loop
                for(BasicBlock * bb : L->getBlocks())
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end();
                        inst != inst_end; inst++)
                    {
                        
                    }
                }
            }
        }
        */

        //We use this map to collect graph of each function, and walk through main() to build the whole CFG
        std::unordered_map<Function *, Seq_Graph*> Func_SG_map;
        std::unordered_map<Function *, DAG*> Func_DAG_map;
        //NOTE: When we have a function, the first argument(int32) n means the following n arguments
        //are output values(also input values)

        //Walk to generate SQ
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);
            if(caller_func == nullptr)
            {
                errs()<<"Fail to get the caller func\n";
                exit(1);
            }
            if(caller_func->size() == 0)
            {
                //errs()<<"Func "<<caller_func->getName()<<" size is 0\n";
                continue;
            }
            
            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;
            else if(target_func_map.find(caller_func) == target_func_map.end()) continue;
            //TO.DO.: Make all target func begin with some prefix, so we can efficiently add more target func
            
            //We only care about the control flow in a function, we dont care the whole CFG
            //So we have a general garph in a function, but a sub-graph representing a loop / branch
            //is also a part of the general graph.
            
            Seq_Graph * SG;
            DAG * dag;
            if(Func_SG_map.find(caller_func) != Func_SG_map.end() && Func_DAG_map.find(caller_func) != Func_DAG_map.end()) 
            {
                continue;
            }
            else
            {
                SG = new Seq_Graph();
                dag = new DAG();
                Func_SG_map[caller_func] = SG;
                Func_DAG_map[caller_func] = dag;
            } 

            size_t inst_counter = 0;
            size_t kernel_counter = 0;
            //std::string func_name = func->getName().str();
            for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                {
                    inst_counter++;
                    if(isa<CallInst>(inst))
                    {
                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                        Function * called_func = call_inst->getCalledFunction();
                        if(called_func == nullptr)
                        {
                            continue;
                        }
                        std::string called_func_name = called_func->getName().str();
                        if(called_func_name == "hipLaunchKernel")
                        {
                            //get the first argument as target kernel function
                            //errs()<<"We get one kernel call inst\n";
                            Value * v = dyn_cast<Value>(call_inst->getArgOperand(0));
                            //errs()<<*v<<"\n";
                            Function * kernel_func = dyn_cast<Function>(v->stripPointerCasts());
                            //errs()<<kernel_func->getName();
                            //errs()<<"\n";

                            //If it's in the kernel function IR, it will call itself(only with the same name of kernel function) again
                            //So we need CallInst as identifier.
                            //QUES.: Should we do more about this?
                            GPUFuncNode * node = new GPUFuncNode(call_inst,kernel_func,true,true);
                            node->CollectBBs(M);

                            node->SetStream(stream_var_ptrs[kernel_counter % 10]);

                            node->dump_inst();
                            node->dumpBBs();
                            errs()<<"***************\n";
                            
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);

                            std::vector<Value*> input_args = GetKernelArgs(node,M);             //Here what we got is the original value of each arguments
                            if(input_args.size() == 0)
                            {
                                std::cout<<"Cannot get input arguments of kernel "<<kernel_func->getName().str()<<std::endl;
                                exit(1);
                            }

                            //Register Input/Output Value into node
                            Value * output_value_n = input_args[0];
                            if(ConstantInt * CI = dyn_cast<ConstantInt>(output_value_n))
                            {
                                size_t output_n = CI->getZExtValue();
                                size_t n = kernel_func->arg_size();
                                errs()<<kernel_func->getName()<<" has "<<n<<" input arguments and "<<output_n<<" output arguments \n";
                                //We dont get the counter argument
                                //QUES.: Would it be good to record the counter argument?
                                for(size_t i = 1; i < 1+output_n; i++)
                                {
                                    node->add_output_value(input_args[i]);
                                }
                                for(size_t i = 1+output_n; i < n; i++)
                                {
                                    //We record the output_counter, but we need to ignore it when construct input_val in DAG
                                    node->add_input_value(input_args[i]);
                                }
                            }
                            else
                            {
                                errs()<<"1st argument of Function "<<called_func_name<<" is not constant value\n";
                                exit(1);
                            }
                            
                            kernel_counter++;

                        }
                        else if(called_func_name.find("cpu_") != std::string::npos)
                        {
                            //This is a cpu function
                            Function * cpu_func = dyn_cast<Function>(call_inst->getFunction());
                            //TO.DO.: Hard to grep the true original input value
                            CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_func,false,true);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else if(called_func_name == "hipMemcpy")
                        {
                            //This is memcpy
                            Function * memcpy_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            MemcpyNode * node = new MemcpyNode(call_inst,memcpy_func,false,false);             //QUES.: Should the gpu_flag be true?
                            node->CollectBBs(M);

                            node->dump_inst();
                            node->dumpBBs();
                            errs()<<"***************\n";
                            
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else
                        {
                            //These are considered to be cpu function(how about hipblasGemmEx etc.?)
                            //In 12-4, we only consider about all Kernel launch without other instructions
                            //TO.DO.: But hipMalloc should be specified to handle.
                            //Function * cpu_library_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            //errs()<<"we meet "<<cpu_library_func->getName()<<"\n";
                            //CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_library_func,false,false);
                            //Node * prev_node = SG->get_last_Node();
                            //SG->Insert(node,prev_node);
                        }
                    }
                    else
                    {
                        //These instructions will be collected by data-flow analysis of each call_inst
                        //For those instructions might not be collected after all these, maybe it's time to kick off
                        //collect them as InstNode and bundle related ones into one SuperNode
                        //So now we dont have to handle them here.
                    }
                }
            }

            //TO.DO.: Print out the function Seq-Graph                              DONE
            //errs()<<"\n\nThis is Seq_Graph for Function "<<caller_func->getName()<<"\n";
            //SG->Print_allNode();

            //Construct DAG
            //TO.DO.: 
            //dag->dump();
            //errs()<<"Begin to construct DAG from SG\n";
            dag->ConstructFromSeq_Graph(SG);
            //dag->dump();
            dag->levelize();
            //dag->dump_level();

        }

        //TO.DO.: Analysis data input & output of each cpu/gpu function
        //Only passed pointer argument can be modified and affect other functions

        //TO.DO.: Partiton the basic blocks of each node (kernel/memcpy)

        return true;
    }


    std::string Profiling::get_father_dir(std::string file_path)
    {
        std::string res = file_path;
        int n = file_path.length();
        int i = n - 1;
        for(; i >= 0; i--)
        {
            if(file_path[i] == '/') break;
        }
        int file_name_length = n - 1 - i;
        while(file_name_length >0)
        {
            file_name_length--;
            res.pop_back();
        }
        //std::cout<<"Father path is "<<res<<std::endl;
        return res;
    }

    void Profiling::get_tool_library_func(Module & M)
    {
        //TO.DO.: Profiling helper func
        
    }

    void Profiling::set_target_func(Module & M)
    {
        for(auto func = M.getFunctionList().begin(), func_end = M.getFunctionList().end();
            func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            std::string func_name = cur_func->getName().str();
            for(auto target_func_name : target_func_list)
            {
                if(func_name.find(target_func_name) != std::string::npos)
                {
                    target_func_map[cur_func] = true;
                }
            }
        }
    }

    std::vector<Value *> Profiling::GetKernelArgs(Node * node, Module & M)
    {
        CallInst * call_inst = node->getCallInst();
        //Function * called_func = node->getFunction();

        //errs()<<called_func->getName().str()<<":\n";

        std::vector<Value *> res;
        
        Value * input_integrated_value = call_inst->getArgOperand(5);
        for(auto use = input_integrated_value->use_begin(), use_end = input_integrated_value->use_end();
            use != use_end; use++)
        {
            if(Instruction * def_inst = dyn_cast<Instruction>(*use))
            {
                //errs()<<"Passed value in lauchKernel:"<<*def_inst<<"\n";
                std::stack<Value*> st;
                /*
                for(auto argv : def_inst->operand_values())
                {
                    errs()<<*argv<<"\n";
                }
                exit(1);
                */
                Value * struct_addr_v = def_inst->getOperand(0);
                bool get_first_flag = false;
                //Only those insts behind the inst producing input_intergrated_value are we need
                for(auto user = struct_addr_v->user_begin(), user_end = struct_addr_v->user_end();
                    user != user_end; user++)
                {
                    if(Instruction * user_inst = dyn_cast<Instruction>(*user))
                    {
                        if(user_inst == def_inst) continue;
                        if(def_inst->getParent() == user_inst->getParent())
                        {
                            //errs()<<*user_inst<<"\n";
                            //TO.DO.: Treat different if user_inst is bitcast or GEP
                            //TO.DO.: How about passing a struct pointer?
                            if(isa<GetElementPtrInst>(*user_inst))
                            {
                                //QUES.: Will we need to handle the first argument here?
                                Value * GEP_Value = dyn_cast<Value>(user_inst);
                                Instruction * bitcast_inst = dyn_cast<Instruction>(*GEP_Value->user_begin());
                                if(!isa<BitCastInst>(bitcast_inst))
                                {
                                    errs()<<"Not a bitcast inst!\n";
                                    exit(1);
                                }
                                Value * bitcast_v = dyn_cast<Value>(bitcast_inst);
                                Instruction * contain_inst = dyn_cast<Instruction>(*bitcast_v->user_begin());
                                if(!isa<StoreInst>(contain_inst))
                                {
                                    errs()<<"Not a store inst(container_inst)\n";
                                    exit(1);
                                }
                                Value * container_v = dyn_cast<Value>(contain_inst->getOperand(0));
                                Instruction * store_inst = dyn_cast<Instruction>(*container_v->user_begin());
                                if(!isa<StoreInst>(store_inst))
                                {
                                    errs()<<"Not a store inst\n";
                                    exit(1);
                                }
                                Value * store_v = dyn_cast<Value>(store_inst->getOperand(0));
                                for(auto st_v_user = store_v->user_begin(), st_v_user_end = store_v->user_end();
                                    st_v_user != st_v_user_end; st_v_user++)
                                {
                                    Instruction * inst = dyn_cast<Instruction>(*st_v_user);
                                    if(isa<StoreInst>(*inst) && inst != store_inst)
                                    //Need to avoid the bitcast inst
                                    {
                                        Value * true_input_data = inst->getOperand(0);
                                        st.push(true_input_data);
                                        //errs()<<"true data from GEP: "<<*true_input_data<<"\n";
                                    }
                                }
                            }
                            else if(isa<BitCastInst>(*user_inst) && !get_first_flag)
                            {
                                //we use this to locate the first argument
                                //We wont run into branch more than once
                                get_first_flag = true;
                                Instruction * bitcast_inst = dyn_cast<Instruction>(user_inst);
                                Value * bitcast_v = dyn_cast<Value>(bitcast_inst);
                                Instruction * contain_inst = dyn_cast<Instruction>(*bitcast_v->user_begin());
                                Value * container = contain_inst->getOperand(0);
                                Instruction * store_inst = dyn_cast<Instruction>(*(++container->user_begin()));
                                Value * true_input_data = store_inst->getOperand(0);
                                st.push(true_input_data);
                                //errs()<<"true data from bitcast: "<<*true_input_data<<"\n";
                            }
                        }
                    }
                }
                while(!st.empty())
                {
                    //Here we should only care about pointer input, others will be copied so that no comflict 
                    //But we use counter outside, so we delay this until we are handling inside DAG
                    Value * cur_v = st.top();
                    //Type * type = cur_v->getType();
                    st.pop();
                    //if(type->isPointerTy()) 
                    res.push_back(cur_v);
                }
            }
        }
        //errs()<<"The input operand of "<<*call_inst<<" are:\n";
        //for(auto arg : res) errs()<<*arg<<"\n";
        return res;
    }

}

char Profiling::ID = 0;

static RegisterPass<Profiling> X("profiling","A Pass to Profile and run data analysis",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new Profiling());});













