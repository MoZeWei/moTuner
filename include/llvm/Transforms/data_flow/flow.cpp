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
#include "llvm/IR/GlobalValue.h"
#include<unordered_map>
#include <limits>
#include<queue>
using namespace llvm;

namespace{
    struct dataflow : public ModulePass{
        static char ID;

        dataflow();
        ~dataflow(){};

        bool runOnModule(Module & M) override;
        void dfs(Value *,Function *, int,std::unordered_map<Value*,bool>&);
        void jump_out_of_parent(Value *, Function *, int,std::unordered_map<Value*,bool>&);
        template<typename T>bool is_in_list(std::vector<T>,T);

        int gemmex_id;
        std::unordered_map<int,Instruction*> gemm_id_call_inst_map;
        std::unordered_map<Instruction*,int> gemm_call_inst_int_map;
        std::unordered_map<int,std::vector<int>> related_gemm_id_map;
        //below one is for general call inst, above are for gemm
        std::vector<CallInst*> call_inst_list;
        std::string modify_matrix_mem_func_name;

    };

    dataflow::dataflow() : ModulePass(ID){
        gemmex_id = 0;
        modify_matrix_mem_func_name = "dududu";
    }

    template<typename T>
    bool dataflow::is_in_list(std::vector<T> l, T target)
    {
        for(T e: l)
        {
            if(target == e) return true;
        }
        return false;
    }

    //NOTE: The related_gemm_id_map we get here contains all related gemm without consindering 
    //1)whether is runned acutally      2) whether it's runned before it(we only consider the before gemm every time we optimize)
    void dataflow::dfs(Value * called_arg, Function * caller_func, int target_id, std::unordered_map<Value*,bool> & dfsed_value_map)
    {
        //TO.DO.: How to avoid repeating the same instruction?                      //DONE
        if(dfsed_value_map.find(called_arg) != dfsed_value_map.end() &&  dfsed_value_map[called_arg]) return;
        else
            dfsed_value_map[called_arg] = true;
        //errs()<<"Current target arg is "<<*called_arg<<"\n";
        //"user" means that this argument is used as argument/operand somewhere
        for(auto user = called_arg->user_begin(), user_end = called_arg->user_end();
            user != user_end; user++)           //User means that this arg is used as operand in these instructions
        {
            //If this arg is just right as the argument of GemmEx
            if(Instruction * inst = dyn_cast<Instruction>(*user))
            {
                //errs()<<"user: "<<*inst<<"\n";
                //errs()<<"Facing instruction of "<<*inst<<"\n";
                if(isa<CallInst>(inst))
                {
                    CallInst * call_inst = dyn_cast<CallInst>(inst);
                    Function * called_func = call_inst->getCalledFunction();
                    if(called_func && called_func->getName() == "hipblasGemmEx")
                    {
                        int cur_id = gemm_call_inst_int_map[inst];
                        //errs()<<"We found the call_inst of GemmEx: "<<*call_inst<<"\n";
                        //we only care about those gemmex that accept this arg as output
                        //NOTE: This makes us wont add gemmex itself into its dependence list
                        if(called_arg==call_inst->getOperand(14))
                        {
                            //TO.DO.: Avoid searching the same gemm                        //DONE
                            if(is_in_list<int>(related_gemm_id_map[target_id],cur_id))
                            {
                                //do nothing
                                //errs()<<"We occur the same GemmEx with id "<<cur_id<<"\n";
                            }
                            else
                            {
                                //errs()<<"The "<<target_id<<"th GemmEx depends on "<<cur_id<<"th GemmEx\n";
                                related_gemm_id_map[target_id].push_back(cur_id);
                                //dfs(call_inst->getOperand(7),caller_func, cur_id, dfsed_value_map);
                                //dfs(call_inst->getOperand(10),caller_func, cur_id, dfsed_value_map);
                                //dfs(call_inst->getOperand(14),caller_func, cur_id, dfsed_value_map);
                            }
                        }
                        else
                            continue;

                    }
                    else if(called_func && called_func->getName() == modify_matrix_mem_func_name)
                    {
                        //TO.DO.: when we run into something like ReadFile() that can modify the Matrix to be a new data matrix
                        //what should we do?
                        
                    }
                    else if(called_func)
                    {
                        //we dont care about other functions
                        //10-26:But if we met a function containing the GemmEx, we wont dig in. 
                        //Oppositely, it will start from the contained GemmEx and jump out to find this GemmEx
                        //QUES.: But in this way, we cannot find the dependency from called func to current GemmEx
                        //like {testfunc(AAC),GemmEx(ABC)} we cannot know the GemmEx depends on the one in testfunc
                        //we only can know testfunc depends on GemmEx's C

                        //find the corresponding parameter
                        //errs()<<"We are facing "<<*call_inst<<"\n";
                        
                        Function * next_caller_func = call_inst->getCalledFunction();       //This is the same as called_func def before
                        //errs()<<next_caller_func->getName()<<"\n";
                        int call_op_num = call_inst->getNumArgOperands();                            //NOTE: Use arg_size instead of getNumOperands
                        int called_func_arg_num = next_caller_func->arg_size();
                        //errs()<<"The called func is "<<next_caller_func->getName().str()<<"\n";
                        //errs()<<"call_inst Operand number is "<<call_op_num<<"\n";
                        //errs()<<"called func argument number is"<<called_func_arg_num<<"\n";
                        if(call_op_num != called_func_arg_num) continue;                    //NOTE: We dont support size-changeable function now(like printf)
                        for(int i = 0; i < call_op_num; i++)
                        {
                            if(called_arg == call_inst->getArgOperand(i))
                            {
                                /*
                                if(called_arg->getType()->isIntegerTy()) exit(1);
                                std::cout<<"Target id is "<<target_id<<"\n";
                                std::cout<<"The called func is "<<next_caller_func->getName().str()<<"\n";
                                std::cout<<"The "<<i<<"th operand is matched\n";
                                std::cout<<"matched passed-in ArgOperand is "<<called_arg->getName().str()<<"\n";
                                std::cout<<"matched Function Argument is "<<next_caller_func->getArg(i)->getName().str()<<"\n";
                                */
                                Value * next_called_arg = next_caller_func->getArg(i);
                                //errs()<<*called_arg<<" -> "<<*next_called_arg<<" of func "<<next_caller_func->getName()<<"\n";;
                                dfs(next_called_arg,next_caller_func,target_id,dfsed_value_map);
                            }
                        }
                        
                        //TO.DO.: Why would this populate the chain?
                        
                        //continue;
                    }
                }
                else
                {
                    //we assume we only have load/store in this branch
                    //errs()<<"Now we have met the load/store inst\n";
                    Value * ret_v = dynamic_cast<Value*>(inst);
                    //errs()<<"inst we expect "<<*ret_v<<"\n";
                    //errs()<<*called_arg<<" ret> "<<*ret_v<<"\n";
                    dfs(ret_v,caller_func, target_id, dfsed_value_map);
                }
            }
        }
        //"use" means that this argument is def/not as an operand somewhere.
        for(auto use = called_arg->use_begin(), use_end = called_arg->use_end(); use != use_end; use++)
        {
            if(Instruction * inst = dyn_cast<Instruction>(*use))
            {
                //errs()<<"use: "<<*inst<<"\n";
                for(auto i = 0; i < inst->getNumOperands(); i++)
                {
                    Value * related_v = inst->getOperand(i);
                    Type * ty = related_v->getType();
                    //errs()<<i<<"th operand "<<*related_v<<" of inst "<<*inst<<"\n";
                    if(ty->isIntegerTy()) continue;
                    if(related_v == called_arg) continue;
                    else 
                    {
                        //errs()<<*called_arg<<" => "<<*related_v<<"\n";
                        dfs(related_v,caller_func, target_id, dfsed_value_map);
                    }
                }
            }
        }

        //For those whose argument is passed through the arguments of parent functions
        //In fact, we should check whether it is in parent's arguments whenever we are handling a new Value
        //So that we can jump out of parent function, get the all coresponding call_inst of parent function
        //and dfs on the coresponding passed arguments of call_inst
        //TO.DO.: 
        //jump_out_of_parent(called_arg,caller_func,target_id,dfsed_value_map);
        //NOTE: We should only jump out form the gemmex function, Avoiding jumping into some common functional functions
    }

    void dataflow::jump_out_of_parent(Value * target_arg, Function * parent_func, int target_id, std::unordered_map<Value*,bool>& dfsed_value_map)
    {
        //errs()<<parent_func->getName()<<"\n";
        /*
        if(parent_func->getName().str() == "main") return;
        if(parent_func->getName().str() == "_Z8testfuncPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_t")
        {
            errs()<<"jumping out of testfunc\n";
            errs()<<"It has total "<<parent_func->arg_size()<<" arguments\n";
        }
        */
        //NOTE: We cannot use getNumOperands to get the argument list size of a function def
        for(size_t i = 0; i < parent_func->arg_size(); i++)
        {
            Value * arg = parent_func->getArg(i);
            //errs()<<"The function "<<parent_func->getName().str()<<" "<<i<<"th argument is "<<*arg<<"\n";
            if(target_arg == arg)
            {
                //TO.DO.: Loop over call_inst_list and locate all corresponding passed-in argument, dfs on them
                for(int j = 0; j < call_inst_list.size(); j++)
                {
                    Function * called_func = call_inst_list[j]->getCalledFunction();
                    if(called_func && called_func == parent_func)
                    {
                        //errs()<<"We now jump out of test func\n";
                        //errs()<<"Parent's "<<i<<"th argument is matched\n";
                        CallInst * call_inst = call_inst_list[j];
                        Value * target_passed_arg = call_inst->getArgOperand(i);
                        Function * new_parent_func = call_inst->getParent()->getParent();
                        //errs()<<*target_arg<<" ^> "<<*target_passed_arg<<"\n";
                        dfs(target_passed_arg,new_parent_func,target_id, dfsed_value_map);
                    }
                }
            }
        }
    }

    bool dataflow::runOnModule(Module &M)
    {
        //TO.DO.: In official version, we should only care about functions outside the tool_library
        //NOTE: We only collect all id of call_inst and gemmex here. Because in dfs, no any in-order ensured
        for(auto func = M.getFunctionList().begin(), end_func = M.getFunctionList().end();
            func != end_func; func++)
        {
            //errs()<<"Now we are facing declare of function "<<func->getName()<<"\n";
            for(auto bb = func->begin(); bb != func->end(); bb++)
            {
                for(auto inst = bb->begin(); inst != bb->end(); inst++)
                {
                    if(CallInst * call_inst = dyn_cast<CallInst>(inst))
                    {
                        call_inst_list.push_back(call_inst);
                        Function * called_func = call_inst->getCalledFunction();
                        if(called_func && called_func->getName() == "hipblasGemmEx")
                        {
                            errs()<<"We get the "<<++gemmex_id<<"th called GemmEx function in "<<*call_inst<<"\n";
                            gemm_id_call_inst_map[gemmex_id]=call_inst;
                            gemm_call_inst_int_map[call_inst] = gemmex_id;
                        }
                    }
                }
            }
        }

        errs()<<"Now we finish collecting all call_inst(including gemmex)\n";

        //Now we are dealing with related-gemm 
        gemmex_id = 0;
        for(auto func = M.getFunctionList().begin(), end_func = M.getFunctionList().end();
            func != end_func; func++)
        {
            //errs()<<"Now we are facing declare of function "<<func->getName()<<"\n";
            for(auto bb = func->begin(); bb != func->end(); bb++)
            {
                for(auto inst = bb->begin(); inst != bb->end(); inst++)
                {
                    //only focus on GemmEx
                    if(CallInst * call_inst = dyn_cast<CallInst>(inst))
                    {
                        Function * called_func = call_inst->getCalledFunction();
                        if(called_func && called_func->getName() == "hipblasGemmEx")
                        {
                            gemmex_id++;
                            errs()<<"Now we use "<<gemmex_id<<"th GemmEx as target GemmEx\n";
                            //get MatrixA argument defined before.
                            Value * Matrix1_argv = call_inst->getArgOperand(7);
                            //For those who has a load / related operation previous
                            Function * caller_func = dyn_cast<Function>(func);
                            std::unordered_map<Value*,bool> dfsed_value_map;
                            dfs(Matrix1_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix1_argv,caller_func,gemmex_id,dfsed_value_map);

                            Value * Matrix2_argv = call_inst->getArgOperand(10);
                            dfsed_value_map.clear();
                            dfs(Matrix2_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix2_argv,caller_func,gemmex_id,dfsed_value_map);

                            Value * Matrix3_argv = call_inst->getArgOperand(14);
                            dfsed_value_map.clear();
                            dfs(Matrix3_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix3_argv,caller_func,gemmex_id,dfsed_value_map);

                        }
                    }
                }
            }
        }

        //running above, we are able to get the whole data-related gemm
        //filter those gemm are not actually runned or not runned before target-gemm
        for(auto it = related_gemm_id_map.begin(); it != related_gemm_id_map.end(); it++)
        {
            std::cout<<"The related GemmEx id of "<<it->first<<" contains: ";
            for(auto id : it->second) std::cout<<id<<" ";
            std::cout<<std::endl;
        }


        return false;
    }

}

char dataflow::ID = 0;
static RegisterPass<dataflow> X("dataflow","A Pass to analysis data flow",
                                false, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new dataflow());});