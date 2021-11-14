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
using namespace llvm;

namespace{
    struct TUNING : public ModulePass{
        static char ID;

        TUNING();
        ~TUNING() {};                                       //Must define this function

        bool runOnModule(Module & M) override;
        bool read_runned_funcID_file();                    //TO.DO.: What result should I get?                                                                  //DONE
        //bool gen_optimized_data_file();
        bool read_optimized_data_file();                    //process the file and get the best parameter combination of each optimized Gemm Function
        bool make_runned_func_ID_set();                     //When reading runned_funcID_file, if meet repeated id, call this func to re-organize the runned_funcID_file
                                                            //next time we wont run into repeated runned func id
        bool get_biggest_dimension();
        //bool is_in_list(int,std::vector<int>);
        void get_tool_library_func(Module & , std::unordered_map<Function*,int>& );
        std::string get_father_path(std::string);
        bool read_last_pass_seperate_error();
        void flush_file(std::string);
        void best_arguments_repair(Module & M);
        void run_data_flow_analysis(Module & M);
        void dataflow_dfs(Value *,Function *, int,std::unordered_map<Value*,bool>&);
        void jump_out_of_parent(Value *, Function *, int,std::unordered_map<Value*,bool>&);
        template<typename T>bool is_in_list(std::vector<T>,T);
        void adjust_related_gemm_map();
        void repair_dfs(int,std::unordered_map<int,bool>&);

        //data
        //Below is for TUNING itself
        std::string runned_funcID_file_path;                //How to setup the file path for each task: Using the name of Module?
        std::string optimized_data_file_path;
        std::string dimension_file_path;
        std::string last_pass_seperate_error_file_path;
        std::vector<int> runned_funcID_list;                     //in-order list
        std::vector<int> to_repair_funcID_list;                   //This is for those current func with too much error
        std::unordered_map<int,std::vector<int>> best_arguments;
        std::unordered_map<Function*,int> tool_func_map;
        
        float max_err_threshold;                            //TO.DO.: Use Command line to set these two value
        float avg_err_threshold;
        int pool_m;
        int pool_n;
        int pool_k;
        int last_tuned_id;

        //below is for data flow
        std::unordered_map<int,Instruction*> gemm_id_call_inst_map;
        std::unordered_map<Instruction*,int> gemm_call_inst_int_map;
        std::unordered_map<int,std::vector<int>> related_gemm_id_map;
        //below one is for general call inst used in data flow, above are for gemm used in data flow
        std::vector<CallInst*> call_inst_list;
        std::string modify_matrix_mem_func_name;                    //TO.DO.: Pay attention to Readfile()
    
    };

    TUNING::TUNING() : ModulePass(ID)
    {
        max_err_threshold = 1.0;
        avg_err_threshold = 0.1;
        last_tuned_id = -1;                                 //To check when first tuning
    }

    bool TUNING::read_runned_funcID_file()
    {
        std::fstream fs(runned_funcID_file_path,std::ios::in);
        if(!fs)
        {
            std::cout<<"Cannot open the runned_funcID_file"<<std::endl;
            exit(1);
        }
        else
        {
            int func_id;
            std::unordered_map<int,bool> m;
            bool reorg_flag = false;
            while(fs>>func_id)
            {
                if(m.find(func_id) != m.end())
                {
                    reorg_flag = true;
                    continue;
                }
                else
                {
                    runned_funcID_list.push_back(func_id);
                    m[func_id] = true;
                }
            }
            fs.close();
            if(reorg_flag) return make_runned_func_ID_set();
            return true;
        }

    }

    //bool TUNING::gen_optimized_data_file()                                  //I dont think this should be here. when record optimized data at the first time,
                                                                            //use read_file to test, if not exist, use ios::out to create one, if exist, use ios::app to output
    /*
    {
        std::fstream fs(optimized_data_file_path,std::ios::out);
        if(!fs)
        {
            std::cout<<"Cannot open the optimized_data_file"<<std::endl;
            exit(1);
        }
        else
        {
            fs<<"";
            fs.close();
            return true;
        }
    }
    */

    bool TUNING::read_optimized_data_file()
    {
        //NOTE.: We only have data about one func in this file
        std::fstream fs(optimized_data_file_path,std::ios::in);
        if(!fs)
        {
            std::cout<<"It's not there, need to create one"<<std::endl;
            std::fstream cf(optimized_data_file_path,std::ios::out);
            if(!cf)
            {
                std::cout<<"Cannot create new optimized data file\n";
                exit(1);
            }
            cf<<"";
            cf.close();
            return true;
        }
        else
        {
            //get the avg time/error of func with the same id
            std::unordered_map<std::string,std::vector<float>> data;
            //data: {"id_p1_p2_p3":[running_time,avg_err,max_err,times]}
            std::unordered_map<int,std::vector<std::string>> funcid_keys;       //store all keys corresponding to one func(used when getting best arguments for each func)
        

            int id;
            int p[3];                           //parameter
            float running_time,avg_err,max_err;
            while(fs>>id>>p[0]>>p[1]>>p[2]>>running_time>>avg_err>>max_err)
            {
                //get the key in dict data
                std::string key = std::to_string(id);
                for(int i = 0; i < 3; i++)              //later we may have five parameter
                {
                    key += '_';
                    key += std::to_string(p[i]);
                }
                funcid_keys[id].push_back(key);
                if(data.find(key) != data.end())
                {
                    float total_running_time = data[key][0] * data[key][3];
                    float total_avg_err = data[key][1] * data[key][3];
                    float total_max_err = data[key][2];
                    data[key][3]++;
                    //QUES.: Should these be all avg values or all max values?
                    data[key][0] = (total_running_time + running_time)/data[key][3];
                    data[key][1] = (total_avg_err + avg_err)/data[key][3];
                    data[key][2] = max_err > total_max_err ? max_err : total_max_err;
                }
                else
                {
                    data[key] = {running_time,avg_err,max_err,1};
                }
            }
            fs.close();

            //Get the best arguments of each ID in data and record into best_arguments
            //TO.DO.: We just need to iterate through the map called "data"
            for(int runned_func_id : runned_funcID_list)
            {
                //here we only care about the func were replaced before and try argument combination 
                if(funcid_keys.find(runned_func_id)==funcid_keys.end()) continue;

                last_tuned_id = runned_func_id;                                             //We only care about the last one tunned func-id in optimized data file

                std::string best_id_argument = "";
                float best_running_time = std::numeric_limits<float>::max();
                for(std::string key : funcid_keys[runned_func_id])
                {
                    if(data[key][1] <= avg_err_threshold && data[key][2] <= max_err_threshold)
                    //TO.DO.: If the error of fp32/fp32(not replaced by faster_mul) is still too high
                    //What should we do?
                    //Solu.: 1. How about flushing the prev best argument back to {1,1,0} which are before
                    //the fore-first gemm func with 1,1,x/-1,-1,-1.(QUES.: How to determine how many pre func's arg to flush?)
                    //How about all prev func goes to 1,1,0? (or maybe we can just flush the data-related prev GEMM?)
                    //2. We can use 0.5 * threshold as official threshold, to ensure the loss will not explode too soon
                    {
                        //NOTE: data of fp32 will be recored like id_-1_-1_-1
                        //Its both errors are 0, so in every situation, performance of fp32 will be used to compare
                        if(best_running_time > data[key][0])
                        {
                            best_running_time = data[key][0];
                            best_id_argument = key;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else
                    {
                        continue;
                    }
                }

                if(best_id_argument.length() == 0)
                {
                    //it means this func will cause too much error under prev best_argument combination
                    //if this is the first gemm, it should just become fp32/fp32 gemm
                    //if not, re-adjust it to be (1,1,0), later should upgrade level of argument-combination of all related gemm
                    //Above is for the last one tuning result, about the one before last-one tuning result, we cannot get
                    //its error in optimized data file, we just can get it from single pass error file.
                    to_repair_funcID_list.push_back(runned_func_id);
                    std::cout<<"The tuned func with id "<<runned_func_id<<" needs to be repaired."<<std::endl;
                    //The best_argument of last tunned gemm func doesn't matter 
                    //because for last-tuned gemm, the first upgrade is to be 1,1,0 anyway(except for the first gemm)
                    //this is left for later process, but need to use last_tuned_id to specify it.
                    //If it's the first gemm, it just need to be set to -1-1-1
                    best_arguments[runned_func_id] = {0,0,0};
                    continue;
                }

                //parse the best_id_argument to fullfill the best_arguments dict
                //id_p1_p2_p3
                std::cout<<"The best argument for func "<<runned_func_id<<" is "<<best_id_argument<<std::endl;      //debug
                std::string delimiter = "_";
                size_t pos = 0;
                std::string part_num;
                std::vector<int> tmp;
                while((pos = best_id_argument.find(delimiter)) != std::string::npos)
                {
                    part_num = best_id_argument.substr(0,pos);
                    //std::cout<<part_num<<" ";                                                               //debug
                    tmp.push_back(std::stoi(part_num));                                                     
                    best_id_argument.erase(0,pos+delimiter.length());
                }
                //std::cout<<best_id_argument<<std::endl;                                                     //debug
                best_arguments[tmp[0]] = {tmp[1],tmp[2],std::stoi(best_id_argument)};
            }

            //TO.DO.: we flush the optimized data file here so that every time we read this file, it only contains                  //DONE
            //tuning info about one gemm.
            flush_file(optimized_data_file_path);

            return true;
        }
        return false;
    }

    bool TUNING::make_runned_func_ID_set()
    {
        //we now have runned_funcID_list, it's in order.
        //so just need to write it to the runned_funcID_file_path
        std::ofstream fs(runned_funcID_file_path);       
        if(!fs)
        {
            std::cout<<"Cannot open the runned_funcID file"<<std::endl;
            exit(1);
        }
        else
        {
            for(int id : runned_funcID_list)
            {
                fs<<id<<std::endl;
            }
            fs.close();
            return true;
        }
        return false;
    }

    bool TUNING::get_biggest_dimension()
    {
        //TO.DO.: read the dimension_file and get the biggest pool size             //Done
        std::fstream fs(dimension_file_path,std::ios::in);
        if(!fs)
        {
            errs()<<"Cannot open the dimension file\n";
            exit(1);
        }
        int m,n,k;
        //TO.DO.: Every variable about pool_size should be size_t type
        size_t max_size = 0;
        //update the pool_m/n/k
        while(fs>>m>>n>>k)
        {
            size_t cur_pool_size = sizeof(float)*(
                                2*m*k +                                    
                                2*k*n +                                    
                                m*n*2) +                                    
                                sizeof(short)*(
                                m*k*2 +                                    
                                k*n*2 +                                    
                                m*n*2);
            if(cur_pool_size > max_size)
            {
                pool_m = m;
                pool_n = n;
                pool_k = k;
                max_size = cur_pool_size;
            }
        }
        fs.close();
        return true;
    }

    bool TUNING::read_last_pass_seperate_error()
    {
        //TO.DO.:                                                           //DONE
        //read it and then flush it.
        std::fstream fs(last_pass_seperate_error_file_path,std::ios::in);
        if(!fs)
        {
            std::cout<<"Cannot open the last pass error file"<<std::endl;
            //TO.DO.: Fucking build one
            std::fstream new_fs(last_pass_seperate_error_file_path,std::ios::out);
            new_fs<<"";
            new_fs.close();
            fs.open(last_pass_seperate_error_file_path,std::ios::in);
        }
        int id,p[3];
        float running_time,avg_err,max_err;
        std::unordered_map<std::string,std::vector<float>> data;
        //std::unordered_map<int,std::vector<std::string>> funcid_keys; 
        while(fs>>id>>p[0]>>p[1]>>p[2]>>running_time>>avg_err>>max_err)
        {
            std::string key = std::to_string(id);
            for(int e : p)
            {
                key += '_';
                key += std::to_string(e);
            }
            //funcid_keys[id].push_back(key);
            if(data.find(key) != data.end())
            {
                float total_running_time = data[key][0] * data[key][3];
                float total_avg_err = data[key][1] * data[key][3];
                float total_max_err = data[key][2];
                data[key][3]++;
                //QUES.: Should these be all avg values or all max values?
                data[key][0] = (total_running_time + running_time)/data[key][3];
                data[key][1] = (total_avg_err + avg_err)/data[key][3];
                data[key][2] = max_err > total_max_err ? max_err : total_max_err;
            }
            else
            {
                data[key] = {running_time,avg_err,max_err,1};
            }
        }
        fs.close();

        //parse error of each gemm func
        
        for(auto it = data.begin(); it != data.end(); it++)         //each gemm func only have one entry
        {
            std::string delimiter = "_";
            size_t pos = 0;
            std::string part_num;
            std::vector<int> tmp;
            std::string id_args = it->first;                        //"id_p1_p2_p3"
            std::vector<float> cur_record = it->second;             //{running_time,avg_err,max_err,times}
            while((pos = id_args.find(delimiter)) != std::string::npos)
            {
                part_num = id_args.substr(0,pos);
                tmp.push_back(std::stoi(part_num));              
                id_args.erase(0,pos+delimiter.length());
            }
            tmp.push_back(std::stoi(id_args));                      // tmp = {id,p1,p2,p3}

            if(cur_record[1] <= avg_err_threshold && cur_record[2] <= max_err_threshold)
            {
                errs()<<"The "<<tmp[0]<<"th func has chosen args\n";
                best_arguments[tmp[0]] = {tmp[1],tmp[2],tmp[3]};        //save it and let later process to upgrade depending on current arguments
            }
            else
            {
                to_repair_funcID_list.push_back(tmp[0]);
                std::cout<<"The "<<tmp[0]<<"th func needs to repair"<<std::endl;
                best_arguments[tmp[0]] = {tmp[1],tmp[2],tmp[3]};
            }

        }
        


        flush_file(last_pass_seperate_error_file_path);
        return true;
    }

    void TUNING::flush_file(std::string file_path)
    {
        //TO.DO.:                                                                   //DONE
        std::fstream fs(file_path,std::ios::out | std::ios::trunc);
        if(!fs)
        {
            std::cout<<"Cannot open the file: "<<file_path<<std::endl;
            exit(1);
        }
        else
        {
            fs<<"";
            fs.close();
        }
    }

    void TUNING::get_tool_library_func(Module& M, std::unordered_map<Function*,int> &tool_func_map)
    {
        //get all function pointer in tool-library, these function should not be scanned and optimized
        Function * tuning_gemm_func_ptr = M.getFunction("_Z17mzw_tuning_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiPcS3_S_");
        if(tuning_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tunning fuc\n";
            exit(1);
        }
        tool_func_map[tuning_gemm_func_ptr] = 1;

        //following func ptr are only used to check, not to replace
        Function * output_dimension_func_ptr = M.getFunction("_Z25mzw_output_dimension_fileiiiPc");
        if(output_dimension_func_ptr == nullptr)
        {
            errs()<<"Cannot get the od func\n";
            exit(1);
        }
        tool_func_map[output_dimension_func_ptr] = 1;
        Function * output_FuncID_func_ptr = M.getFunction("_Z22mzw_output_FuncID_fileiPc");
        if(output_FuncID_func_ptr == nullptr)
        {
            errs()<<"Cannot get the ofid func\n";
            exit(1);
        }
        tool_func_map[output_FuncID_func_ptr] = 1;
        Function * output_ErrP_func_ptr = M.getFunction("_Z31mzw_output_Err_Performance_fileiiiifffPc");
        if(output_ErrP_func_ptr == nullptr)
        {
            errs()<<"Cannot get the oep func\n";
            exit(1);
        }
        tool_func_map[output_ErrP_func_ptr] = 1;
        Function * checkerr_func_ptr = M.getFunction("_Z23mzw_check_hipblas_error15hipblasStatus_t");
        if(checkerr_func_ptr == nullptr)
        {
            errs()<<"Cannot get the che func\n";
            exit(1);
        }
        tool_func_map[checkerr_func_ptr] = 1;
        Function * wrapper_gemm_func_ptr = M.getFunction("_Z18mzw_wrapper_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiPcS3_S3_");
        if(wrapper_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the wrapper_gemm func\n";
            exit(1);
        }
        tool_func_map[wrapper_gemm_func_ptr] = 1;
        Function * rcrp_func_ptr = M.getFunction("_Z35mzw_Result_Check_Record_PerformancePvS_iiiiiifPci");
        if(rcrp_func_ptr == nullptr)
        {
            errs()<<"Cannot get the rcrp func\n";
            exit(1);
        }
        tool_func_map[rcrp_func_ptr] = 1;
        Function * fastermul_func_ptr = M.getFunction("_Z14mzw_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iii");
        if(fastermul_func_ptr == nullptr)
        {
            errs()<<"Cannot get the faster_mul func\n";
            exit(1);
        }
        tool_func_map[fastermul_func_ptr] = 1;
        Function * readfile_func_ptr = M.getFunction("_Z12mzw_ReadFileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERmPvm");
        if(readfile_func_ptr == nullptr)
        {
            errs()<<"Cannot get the readfile func\n";
            exit(1);
        }
        tool_func_map[readfile_func_ptr] = 1;
        Function * writefile_func_ptr = M.getFunction("_Z13mzw_WriteFileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvm");
        if(writefile_func_ptr == nullptr)
        {
            errs()<<"Cannot get the writefile func\n";
            exit(1);
        }
        tool_func_map[writefile_func_ptr] = 1;
        Function * devicecast_func_ptr = M.getFunction("_Z33__device_stub__mzw_fp32_cast_fp16PfP6__halfii");
        if(devicecast_func_ptr == nullptr)
        {
            errs()<<"Cannot get the device_cast func\n";
            exit(1);
        }
        tool_func_map[devicecast_func_ptr] = 1;
        devicecast_func_ptr = M.getFunction("_Z33__device_stub__mzw_fp16_cast_fp32P6__halfPfii");
        if(devicecast_func_ptr == nullptr)
        {
            errs()<<"Cannot get the device_cast func\n";
            exit(1);
        }
        tool_func_map[devicecast_func_ptr] = 1;
        Function * quan_func_ptr = M.getFunction("_Z8mzw_quanPvS_iiPf17hipblasDatatype_tb");
        if(quan_func_ptr == nullptr)
        {
            errs()<<"Cannot get quan func\n";
            exit(1);
        }
        tool_func_map[quan_func_ptr] = 1;
        Function * dequan_func_ptr = M.getFunction("_Z10mzw_dequanPvS_iiPf17hipblasDatatype_t");
        if(dequan_func_ptr == nullptr)
        {
            errs()<<"Cannot get dequan func\n";
            exit(1);
        }
        tool_func_map[dequan_func_ptr] = 1;  
        Function * checker_faster_mul_func_ptr = M.getFunction("_Z22mzw_checker_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iiiiPcS3_");
        if(checker_faster_mul_func_ptr == nullptr)
        {
            errs()<<"Cannot get checker_faster_mul func\n";
            exit(1);
        }
        tool_func_map[checker_faster_mul_func_ptr] = 1;  
        Function * checker_fp32_gemm_func_ptr = M.getFunction("_Z18mzw_checker_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiiiiPcS3_");
        if(checker_fp32_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get checker_fp32_gemm func\n";
            exit(1);
        }
        tool_func_map[checker_fp32_gemm_func_ptr] = 1;  
    }

    std::string TUNING::get_father_path(std::string file_path)
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
        std::cout<<"Father path is "<<res<<std::endl;
        return res;
    }

    template<typename T>
    bool TUNING::is_in_list(std::vector<T> l, T target)
    {
        for(T e: l)
        {
            if(target == e) return true;
        }
        return false;
    }

    //NOTE: The related_gemm_id_map we get here contains all related gemm without consindering 
    //1)whether is runned acutally      2) whether it's runned before it(we only consider the before gemm every time we optimize)
    void TUNING::dataflow_dfs(Value * called_arg, Function * caller_func, int target_id, std::unordered_map<Value*,bool> & dfsed_value_map)
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
                        if(called_arg==call_inst->getArgOperand(14))
                        {
                            //TO.DO.: Avoid searching the same gemm                        //DONE
                            if( related_gemm_id_map.find(target_id) != related_gemm_id_map.end()
                                &&is_in_list<int>(related_gemm_id_map[target_id],cur_id))
                            {
                                //do nothing
                                //errs()<<"We occur the same GemmEx with id "<<cur_id<<"\n";
                            }
                            else
                            {
                                //errs()<<"The "<<target_id<<"th GemmEx depends on "<<cur_id<<"th GemmEx\n";
                                related_gemm_id_map[target_id].push_back(cur_id);
                                //dataflow_dfs(call_inst->getOperand(7),caller_func, cur_id, dfsed_value_map);
                                //dataflow_dfs(call_inst->getOperand(10),caller_func, cur_id, dfsed_value_map);
                                //dataflow_dfs(call_inst->getOperand(14),caller_func, cur_id, dfsed_value_map);
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
                                dataflow_dfs(next_called_arg,next_caller_func,target_id,dfsed_value_map);
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
                    dataflow_dfs(ret_v,caller_func, target_id, dfsed_value_map);
                }
            }
        }
        //"use" means that this argument is def(not as an operand) somewhere.
        for(auto use = called_arg->use_begin(), use_end = called_arg->use_end(); use != use_end; use++)
        {
            if(Instruction * inst = dyn_cast<Instruction>(*use))
            {
                //errs()<<"use: "<<*inst<<"\n";
                //TO.DO.: If we meet the %s = call func(), ignore it.               DONE
                if(isa<CallInst>(inst)) continue;
                for(auto i = 0; i < inst->getNumOperands(); i++)
                {
                    Value * related_v = inst->getOperand(i);
                    Type * ty = related_v->getType();
                    //errs()<<i<<"th operand "<<*related_v<<" of inst "<<*inst<<"\n";
                    if(ty->isIntegerTy()) continue;                                                                 //Because we may incur a instruction using constant Int to calculate address pointer
                    if(related_v == called_arg) continue;                                                           //This is to avoid the instruction like a = a, but this will be eliminated by compiler
                    else 
                    {
                        //errs()<<*called_arg<<" => "<<*related_v<<"\n";
                        dataflow_dfs(related_v,caller_func, target_id, dfsed_value_map);
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

    //jump_out_of_parent should be alike to the data_dfs but without digging in.
    //so it need its own dfs
    void TUNING::jump_out_of_parent(Value * target_arg, Function * parent_func, int target_id, std::unordered_map<Value*,bool>& dfsed_value_map)
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

        //Expand the related field of called_arg, we care about every related variable whether they
        //corresponding to the arg of caller_func
        //TO.DO.: DFS on all related arg of current target_arg, and compare it with all arg of parent func
        //Once it's matched, jump out!
        //This map-search's target is within current parent func, it will not produce conflict to the one in dataflow_dfs
        if(dfsed_value_map.find(target_arg) != dfsed_value_map.end() && dfsed_value_map[target_arg])    return;
        else dfsed_value_map[target_arg] = true;

        for(auto user = target_arg->user_begin(), user_end = target_arg->user_end(); user != user_end; user++)
        {
            if(Instruction * inst = dyn_cast<CallInst>(*user))
            {
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
                        if(target_arg==call_inst->getArgOperand(14))
                        {
                            //TO.DO.: Avoid searching the same gemm                        //DONE
                            if( related_gemm_id_map.find(target_id) != related_gemm_id_map.end()
                                &&is_in_list<int>(related_gemm_id_map[target_id],cur_id))
                            {
                                //do nothing
                                //errs()<<"We occur the same GemmEx with id "<<cur_id<<"\n";
                            }
                            else
                            {
                                //errs()<<"The "<<target_id<<"th GemmEx depends on "<<cur_id<<"th GemmEx\n";
                                related_gemm_id_map[target_id].push_back(cur_id);
                                //dataflow_dfs(call_inst->getOperand(7),caller_func, cur_id, dfsed_value_map);
                                //dataflow_dfs(call_inst->getOperand(10),caller_func, cur_id, dfsed_value_map);
                                //dataflow_dfs(call_inst->getOperand(14),caller_func, cur_id, dfsed_value_map);
                            }
                        }
                        else
                            continue;
                    }
                    else
                    {
                        //we should not care about other function call
                    }
                }
                else
                {
                    Value * ret_v = dynamic_cast<Value*>(inst);
                    jump_out_of_parent(ret_v,parent_func,target_id,dfsed_value_map);
                }
            }
        }

        //Because we only care about what produces such a value, the defs matter. So we only iterate on use-def chain
        for(auto use = target_arg->use_begin(), use_end = target_arg->use_end(); use != use_end; use++ )
        {
            if(Instruction * inst = dyn_cast<Instruction>(*use))
            {
                for(auto i = 0; i < inst->getNumOperands(); i++)
                {
                    Value * related_v = inst->getOperand(i);
                    Type * ty = related_v->getType();
                    if(ty->isIntegerTy()) continue;
                    if(related_v == target_arg) continue;
                    else
                    {
                        jump_out_of_parent(related_v,parent_func,target_id,dfsed_value_map);
                    }
                }
            }
        }


        //NOTE: We cannot use getNumOperands to get the argument list size of a function def
        //Here we cpmpare the target_arg to the argument signment in the def of current function
        //Once its matched, it can jump out.
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
                        jump_out_of_parent(target_passed_arg,new_parent_func,target_id, dfsed_value_map);
                    }
                }
            }
        }
    }

    void TUNING::run_data_flow_analysis(Module & M)
    {
        int gemmex_id = 0;
        for(auto func = M.getFunctionList().begin(), end_func = M.getFunctionList().end();
            func != end_func; func++)
        {
            //TO.DO.: dont do this on tool_library func
            Function * cur_func = dyn_cast<Function>(func);
            if(tool_func_map.find(cur_func)!=tool_func_map.end()) continue;
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

        //errs()<<"Now we finish collecting all call_inst(including gemmex)\n";

        //Now we are dealing with related-gemm 
        gemmex_id = 0;
        for(auto func = M.getFunctionList().begin(), end_func = M.getFunctionList().end();
            func != end_func; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(tool_func_map.find(cur_func)!=tool_func_map.end()) continue;
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
                            //errs()<<"Now we use "<<gemmex_id<<"th GemmEx as target GemmEx\n";
                            //get MatrixA argument defined before.
                            Value * Matrix1_argv = call_inst->getArgOperand(7);
                            //For those who has a load / related operation previous
                            Function * caller_func = dyn_cast<Function>(func);
                            std::unordered_map<Value*,bool> dfsed_value_map;
                            dataflow_dfs(Matrix1_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix1_argv,caller_func,gemmex_id,dfsed_value_map);

                            Value * Matrix2_argv = call_inst->getArgOperand(10);
                            dfsed_value_map.clear();
                            dataflow_dfs(Matrix2_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix2_argv,caller_func,gemmex_id,dfsed_value_map);

                            Value * Matrix3_argv = call_inst->getArgOperand(14);
                            dfsed_value_map.clear();
                            dataflow_dfs(Matrix3_argv,caller_func,gemmex_id,dfsed_value_map);
                            dfsed_value_map.clear();
                            jump_out_of_parent(Matrix3_argv,caller_func,gemmex_id,dfsed_value_map);

                        }
                    }
                }
            }
        }
    }

    //Delete those not-runned / not forehead GemmEx func of each (Because dataflow_dfs doesn't care about function running order)
    void TUNING::adjust_related_gemm_map()
    {
        for(std::unordered_map<int,std::vector<int>>::iterator it = related_gemm_id_map.begin(),
            end = related_gemm_id_map.end(); it != end; it++)
        {
            int target_id = it->first;
            int target_index = -1;
            for(int i = 0; i < runned_funcID_list.size(); i++)
            {
                if(runned_funcID_list[i] == target_id)
                {
                    target_index = i;
                    break;
                }
            }
            if(target_index == -1) continue;
            std::vector<int> backup = it->second;
            it->second.clear();
            for(auto candidate_id : backup)
            {
                int candidate_index = -1;
                for(int i = 0; i < runned_funcID_list.size(); i++)
                {
                    if(runned_funcID_list[i] == candidate_id)
                    {
                        candidate_index = i;
                        break;
                    }
                }
                if(candidate_index == -1) continue;
                else
                {
                    if(candidate_index < target_index)
                    {
                        it->second.push_back(candidate_id);
                    }
                }
            }
            
        }
    }

    void TUNING::repair_dfs(int to_repair_gemm_id, std::unordered_map<int,bool> & repaired_gemm_id_m)
    {
        if(repaired_gemm_id_m.find(to_repair_gemm_id) != repaired_gemm_id_m.end() &&  repaired_gemm_id_m[to_repair_gemm_id])
        {
            return;
        }
        /*
        else if(to_repair_gemm_id == runned_funcID_list[0])
        {
            //Here we should only do this when we only tune 
            best_arguments[to_repair_gemm_id][0] = -1;
            best_arguments[to_repair_gemm_id][1] = -1;
            best_arguments[to_repair_gemm_id][2] = -1;

            return;
        }
        */
        else
        {
            if(best_arguments[to_repair_gemm_id][0] == 1 && best_arguments[to_repair_gemm_id][1] == 1)
            {
                best_arguments[to_repair_gemm_id][0] = -1;
                best_arguments[to_repair_gemm_id][1] = -1;
                best_arguments[to_repair_gemm_id][2] = -1;
            }
            else if(best_arguments[to_repair_gemm_id][0]== -1 && best_arguments[to_repair_gemm_id][1] == -1
                    && best_arguments[to_repair_gemm_id][2] == -1)
            {
                //do nothing
                return;
            }
            else
            {
                best_arguments[to_repair_gemm_id][0] = 1;
                best_arguments[to_repair_gemm_id][1] = 1;
                best_arguments[to_repair_gemm_id][2] = 0;
            }
            repaired_gemm_id_m[to_repair_gemm_id] = true;
            for(auto next_to_repair_gemm_id : related_gemm_id_map[to_repair_gemm_id])
            {
                repair_dfs(next_to_repair_gemm_id,repaired_gemm_id_m);
            }
        }
    }

    void TUNING::best_arguments_repair(Module & M)
    {
        //Get the graph of data-related gemm(with the map of id and call_inst * )
        //First, run the data flow
        run_data_flow_analysis(M);
        //Second, modify the related-gemm map according to the runned func ID list
        //to ensure that func only depends on the ones running before it.
        adjust_related_gemm_map();
        //check the repaired gemm id list, upgrade their and lated gemms' argument (xxx->110, 11x->-1-1-1, -1-1-1 stays) using bfs on previous graph
        std::unordered_map<int,bool> repaired_gemm_id_m;
        for(auto to_repair_gemm_id : to_repair_funcID_list)
        {
            //TO.DO.: 
            repair_dfs(to_repair_gemm_id,repaired_gemm_id_m);
        }
        errs()<<"*******************************\n";
        for(auto it = related_gemm_id_map.begin(); it != related_gemm_id_map.end(); it++)
        {
            errs()<<"Source gemm "<<it->first;
            errs()<<" Depends on: ";
            for(auto id : it->second) errs()<<id<<" ";
            errs()<<"\n";
        }
        errs()<<"*******************************\n";
    }

    bool TUNING::runOnModule(Module & M)
    {
        std::string ModuleName = M.getModuleIdentifier();
        //TO.DO.: Need to keep the same with the ones in pass 1                 //DONE
        std::string father_path = get_father_path(ModuleName);
        runned_funcID_file_path = father_path+"calledFuncID.txt";
        optimized_data_file_path = father_path+"optimized_data.txt";
        dimension_file_path = father_path+"dimension.txt";
        last_pass_seperate_error_file_path = father_path+"single_pass_err.txt";

        get_tool_library_func(M,tool_func_map);
        read_runned_funcID_file();
        read_optimized_data_file();
        get_biggest_dimension();
        read_last_pass_seperate_error();

        //re-adjust the best_arguments based on repaired_func_id_list
        best_arguments_repair(M);

        //now we have in-order runned func id list and optimized data list
        int optimized_func_num = best_arguments.size();                         //No matter whether a func can gain legel precision loss under any argument combination ot not
                                                                                //best_arguents' size always equal to the number of func in single pass error file + number of func in optimized data file(both flushed each time)
        int next_func_id;
        if(optimized_func_num == runned_funcID_list.size())                     //QUES.: If we dont modify anything, will it break?     ANS.: No
        {
            //The last two pass is to ensure the last precision loss, wont differ from pass before, but we need to skip the tuning_func_replace step
            //we only care about the last func tuning result and single pass error file
            next_func_id = -1;
        }
        else
        {
            next_func_id = runned_funcID_list[optimized_func_num];
            std::cout<<"We have total "<<runned_funcID_list.size()<<" GemmEx to tune, and now we are tuning the "<<next_func_id<<"th GemmEx func"<<std::endl; 
        }

        //errs()<<"Get all functions in tool-library\n";

        Function * tuning_gemm_func_ptr = M.getFunction("_Z17mzw_tuning_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiPcS3_S_");
        if(tuning_gemm_func_ptr==nullptr)
        {
            errs()<<"Cannot get the tuning gemm func ptr\n";
            exit(1);
        }
        FunctionType * tuning_gemm_func_type = tuning_gemm_func_ptr->getFunctionType();

        //TO.DO.: In pass2, we should use checker_faster_mul                                                    //DONE
        /*
        Function * fastermul_func_ptr = M.getFunction("_Z14mzw_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iii");
        if(fastermul_func_ptr==nullptr)
        {
            errs()<<"Cannot get the faster mul func ptr\n";
            exit(1);
        }
        FunctionType * fastermul_func_type = fastermul_func_ptr->getFunctionType();
        */
        Function * fastermul_checker_mul_func_ptr = M.getFunction("_Z22mzw_checker_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iiiiPcS3_");
        if(fastermul_checker_mul_func_ptr == nullptr)
        {
            errs()<<"Cannot get the faster_mul_checker func ptr\n";
            exit(1);
        }
        FunctionType * fastermul_checker_mul_func_type = fastermul_checker_mul_func_ptr->getFunctionType();

        //This is for those gemmex which are checked not suitable for faster_mul
        //they also need to print info on single pass err file
        Function * original_gemm_wrapper_func_ptr = M.getFunction("_Z18mzw_checker_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiiiiPcS3_");
        if(original_gemm_wrapper_func_ptr == nullptr)
        {
            errs()<<"Cannot get the original_gemm_wrapper func ptr\n";
            exit(1);
        }
        FunctionType * original_gemm_wrapper_func_type = original_gemm_wrapper_func_ptr->getFunctionType();

        int gemm_id_tracer = 0;

        //Here we already need to generate a memory pool according to biggest GEMM dimension
        size_t pool_size = 
        sizeof(float)*(
        2*pool_m*pool_k +                                     //for hp_matrix A / RA
        2*pool_k*pool_n +                                     //for hp_matrix B / RB
        pool_m*pool_n*2) +                                    //for hp_matrix C / TMP
        sizeof(unsigned short)*(
        pool_m*pool_k*2 +                                     //for lp_matrix A / RA
        pool_k*pool_n*2 +                                     //for lp_matrix B / RB
        pool_m*pool_n*2);                                     //for lp_matrix C / TMP
        std::cout<<"Pool size: "<<pool_size<<std::endl;

        //locate the main function and add the hipMalloc(memory pool,pool_size) as well as hipFree(memory pool)
        //The memory_pool variable should be global and use hipMalloc in main()
        //We can directly use i8** mempool_ptr
        PointerType * int8PtrType = PointerType::getUnqual(Type::getInt8Ty(M.getContext()));
        if(int8PtrType == nullptr)
        {
            errs()<<"Cannot generate i8* ptr type\n";
            exit(1);
        }
        GlobalVariable * gvar_ptr_mempool = new GlobalVariable(M,int8PtrType,false,GlobalValue::CommonLinkage,
                                                                0,"mzw_mempool");
        gvar_ptr_mempool->setAlignment(MaybeAlign(4));
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(int8PtrType);
        if(NULLPTR == nullptr)
        {
            errs()<<"Cannot generate nullptr value\n";
            exit(1);
        }
        gvar_ptr_mempool->setInitializer(NULLPTR);                                                      //Need to check

        //Now we have a global memory_pool pointer: void * mzw_mempool = nullptr;
        //value representing size of pool will not generate globally.

        std::vector<Instruction*> inst_del;

        //TO.DO.: Build up a map of gemm_id->called_inst

        errs()<<"Now begin to loop over all functions\n";
        //Now we have best arguments for all tuned GemmEx functions
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            end_Func = M.getFunctionList().end(); func != end_Func; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);
            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;                        //now we wont scan the func in tool-library
            //errs()<<"Now we are facing declare of function "<<func->getName()<<"\n";
            
            //In main(), insert hipmalloc at the head and hipFree at the tail
            if(caller_func->getName() == "main")
            {
                int inst_counter = 0;
                Instruction * head_inst;
                std::vector<Instruction*> end_inst_list;        //store all ret instruction in main()
                for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                    {
                        if(!inst_counter) head_inst = dyn_cast<Instruction>(inst);
                        inst_counter++;
                        //If this is a ret inst, store it into end_inst_list
                        if(isa<ReturnInst>(inst))
                        {
                            Instruction * ret_inst = dyn_cast<Instruction>(inst);
                            end_inst_list.push_back(ret_inst);
                        }
                    }
                }

                //For hipMalloc() at the head of main

                IRBuilder<> builder(head_inst);                                                         //QUES.: Should we need to insert before head_inst->getNextNode()
                Function * hipMalloc_func_ptr = M.getFunction("hipMalloc");
                if(hipMalloc_func_ptr == nullptr)
                {
                    errs()<<"Cannot get the hipMalloc func ptr\n";
                    exit(1);
                }
                FunctionType * hipMalloc_func_type = hipMalloc_func_ptr->getFunctionType();
                Function * hipFree_func_ptr = M.getFunction("hipFree");
                if(hipFree_func_ptr == nullptr)
                {
                    errs()<<"Cannot get the hipFree func ptr\n";
                    exit(1);
                }
                FunctionType * hipFree_func_type = hipFree_func_ptr->getFunctionType();

                /*
                //TO.DO.: Create a nested pointer of gvar_ptr_mempool                                   //No need to do
                //insert a line of code hipMalloc((void**)&mzw_mempool,pool_size)
                //In LLVM IR, the nested pointer should be i8 **
                //So first, we need to create a pointertype representing i8**
                PointerType * int8_ptr_type = Type::getInt8PtrTy(M.getContext());
                if(int8_ptr_type == nullptr)
                {
                    errs()<<"Fail to create int8_ptr_type\n";
                    exit(1);
                }
                
                PointerType * nested_int8ptr_type = int8_ptr_type->getPointerTo();
                if(nested_int8ptr_type == nullptr)
                {
                    errs()<<"Fail to create nested int8** type\n";
                    exit(1);
                }
                
                Value * pool_ptr_ptr = builder.CreateBitCast(gvar_ptr_mempool,nested_int8ptr_type,"mzw_pool_ptr_ptr");
                if(!pool_ptr_ptr)
                {
                    errs()<<"Fail to create the pointer to pointer of memory pool\n";
                    exit(1);
                }
                */

                //Here we just use the gvar_ptr_mempool as the first input arg
                Value * pool_ptr_ptr = dynamic_cast<Value*>(gvar_ptr_mempool);

                ConstantInt * MEMPOOLSIZE = builder.getInt64(pool_size);                    //NOTE:@hipMalloc(i8** nonnull %207, i64 1073741824) 不能是int32
                Value * argv_poolsize = dynamic_cast<Value*>(MEMPOOLSIZE);
                if(argv_poolsize == nullptr)
                {
                    errs()<<"Fail to create arg for poolsize\n";
                    exit(1);
                }
                
                std::vector<Value*> args = {pool_ptr_ptr,argv_poolsize};
                if(!builder.CreateCall(hipMalloc_func_type,hipMalloc_func_ptr,makeArrayRef(args)))
                {
                    errs()<<"Cannot create the call inst of hipMalloc\n";
                    exit(1);
                }
                
                while(args.size()) args.pop_back();

                //For hipFree() at the every ret of main
                for(Instruction * ret_inst : end_inst_list)
                {
                    builder.SetInsertPoint(ret_inst);
                    //get the ptr in the gvar_ptr_mempool
                    Value * pool_ptr = builder.CreateLoad(gvar_ptr_mempool,"mzw_pool_ptr");
                    if(pool_ptr == nullptr)
                    {
                        errs()<<"Cannot create load inst for pool_ptr\n";
                        exit(1);
                    }

                    args.push_back(pool_ptr);
                    if(!builder.CreateCall(hipFree_func_type,hipFree_func_ptr,makeArrayRef(args)))
                    {
                        errs()<<"Cannot create the call inst of hipFree\n";
                        exit(1);
                    }
                    args.pop_back();
                }

            }

            for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                {
                    if(isa<CallInst>(inst))
                    {
                        Function * called_func = dyn_cast<CallInst>(inst)->getCalledFunction();
                        if(called_func && called_func->getName() == "hipblasGemmEx")
                        {
                            gemm_id_tracer++;
                            //errs()<<"Now we get the GemmEx\n";
                            
                            //check whether this func id is in the runned_funcid_list or not
                            //If yes, if it's in the best_arguments:
                            //replace the original GemmEx by faster_mul with new arguments
                            //If yes and if it's not in the best_arguments, but equals to next_func_id:
                            //replace it with tuning_gemmex
                            //If yes and if it's not in the best arguments also not equals to next_func_id:
                            //move on
                            //If no, move on
                            if(is_in_list<int>(runned_funcID_list,gemm_id_tracer))
                            {
                                std::cout<<"The "<<gemm_id_tracer<<"th GemmEx is runned"<<std::endl;
                                if(best_arguments.find(gemm_id_tracer) != best_arguments.end())
                                {
                                    //This func of "gemm_id_tracer" is tuned before
                                    //TO.DO.: In the suitation of -1,-1,-1, we cannot replace it by faster_mul, we need to                          //DONE
                                    //cut this out right now, so we need to check the best_argument now
                                    //NOTE.: For those function being repaired but still proved not able to be faster_mul,
                                    //We should replace it with wrapper_original_fp32/fp32_GemmEx
                                    if(best_arguments[gemm_id_tracer][0] == -1)
                                    {
                                        std::cout<<"The "<<gemm_id_tracer<<"th GemmEx is replaced by checker_fp32/fp32_GemmEx"<<std::endl;
                                        //TO.DO.: If we dont replace it, its result wont appear on the single_pass_err_file                         //DONE
                                        //So we need an extra wrapper for it to calculate the gemm in fp32/fp32, also output its arg/err in single_pass_err_file
                                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                                        std::vector<Value*> args;
                                        unsigned int args_num = call_inst->arg_size();
                                        for(auto i = 0; i < args_num; i++)
                                        {
                                            Value * argv = call_inst->getArgOperand(i);
                                            args.push_back(argv);
                                        }
                                        IRBuilder<> builder(dyn_cast<Instruction>(inst));
                                        //Extra args: int Gemm_ID, int p1, int p2, int p3, char * matrix_file_path, char * single_pass_error_file_path
                                        ConstantInt * constantInt_ID = builder.getInt32(gemm_id_tracer);
                                        if(constantInt_ID == nullptr)
                                        {
                                            errs()<<"Cannot get constant value of gemm id\n";
                                            exit(1);
                                        }
                                        Value * argv_id = dynamic_cast<Value*>(constantInt_ID);
                                        args.push_back(argv_id);

                                        for(int i = 0; i < 3; i++)
                                        {
                                            ConstantInt * constantInt_pi = builder.getInt32(best_arguments[gemm_id_tracer][i]);
                                            if(constantInt_pi == nullptr)
                                            {
                                                errs()<<"Cannot get the constant pi\n";
                                                exit(1);
                                            }
                                            Value * argv_pi = dynamic_cast<Value*>(constantInt_pi);
                                            args.push_back(argv_pi);
                                        }

                                        std::string Matrix_File_Name = father_path + "GEMM_" + std::to_string(gemm_id_tracer) + ".bin";
                                        char matrix_file_path_char[200];
                                        strcpy(matrix_file_path_char,Matrix_File_Name.c_str());
                                        Value * argv_matrix_file_path = builder.CreateGlobalStringPtr(matrix_file_path_char);
                                        args.push_back(argv_matrix_file_path);

                                        char single_pass_err_file_path_char[200];
                                        strcpy(single_pass_err_file_path_char,last_pass_seperate_error_file_path.c_str());
                                        Value * argv_single_pass_err_file_path = builder.CreateGlobalStringPtr(single_pass_err_file_path_char);
                                        args.push_back(argv_single_pass_err_file_path);

                                        Value * original_gemm_ret_v = builder.CreateCall(original_gemm_wrapper_func_type,original_gemm_wrapper_func_ptr,makeArrayRef(args));
                                        if(!original_gemm_ret_v)
                                        {
                                            errs()<<"Cannot create call to the wrapper fp32/fp32 GemmEx with a checker\n";
                                            exit(1);
                                        }
                                        call_inst->replaceAllUsesWith(original_gemm_ret_v);

                                        inst_del.push_back(call_inst);
                                    }
                                    else
                                    {
                                        //NOTE.: replaced those GemmEx with faster_mul which has permitted-precision-loss under certain argument combination
                                        std::cout<<"The "<<gemm_id_tracer<<"th GemmEx is replaced by checker_faster_mul"<<std::endl;
                                        //replace the original GemmEx by faster_mul with new arguments
                                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                                        //get all the original args
                                        std::vector<Value*> args;
                                        unsigned int args_num = call_inst->arg_size();
                                        for(auto i = 0; i < args_num; i++)
                                        {
                                            Value * argv = call_inst->getArgOperand(i);
                                            args.push_back(argv);
                                        }
                                        //TO.DO.: generate new args and push it in args                                                             //DONE
                                        //including memory_pool, p1, p2, p3
                                        //memory_pool is the global variable
                                        IRBuilder<> builder(dyn_cast<Instruction>(inst));
                                        //TO.DO.: Extra arguments for faster_mul_checker func : mempool, id, p1,p2,p3, matrix_file_path, single_pass_err_file_path          //DONE
                                        Value * argv_poolptr = builder.CreateLoad(gvar_ptr_mempool,"mzw_pool_ptr");
                                        if(argv_poolptr == nullptr)
                                        {
                                            errs()<<"Cannot get poolptr arg for checker_faster_mul func\n";
                                            exit(1);
                                        }
                                        args.push_back(argv_poolptr);

                                        ConstantInt * constantInt_ID = builder.getInt32(gemm_id_tracer);
                                        if(constantInt_ID == nullptr)
                                        {
                                            errs()<<"Cannot get constant value of gemm id\n";
                                            exit(1);
                                        }
                                        Value * argv_id = dynamic_cast<Value*>(constantInt_ID);
                                        args.push_back(argv_id);

                                        for(int i = 0 ; i < 3; i ++)
                                        {
                                            ConstantInt * constantInt_pi = builder.getInt32(best_arguments[gemm_id_tracer][i]);
                                            if(constantInt_pi == nullptr)
                                            {
                                                errs()<<"Cannot gen int pi for fastermul\n";
                                                exit(1);
                                            }
                                            Value * argv_pi = dynamic_cast<Value*>(constantInt_pi);
                                            args.push_back(argv_pi);
                                        }

                                        std::string Matrix_File_Name = father_path + "GEMM_" + std::to_string(gemm_id_tracer) + ".bin";
                                        char matrix_file_path_char[200];
                                        strcpy(matrix_file_path_char,Matrix_File_Name.c_str());
                                        Value * argv_matrix_file_path = builder.CreateGlobalStringPtr(matrix_file_path_char);
                                        args.push_back(argv_matrix_file_path);

                                        char single_pass_err_file_path_char[200];
                                        strcpy(single_pass_err_file_path_char,last_pass_seperate_error_file_path.c_str());
                                        Value * argv_single_pass_err_file_path = builder.CreateGlobalStringPtr(single_pass_err_file_path_char);
                                        args.push_back(argv_single_pass_err_file_path);

                                        Value * fastermul_ret_v = builder.CreateCall(fastermul_checker_mul_func_type,fastermul_checker_mul_func_ptr,makeArrayRef(args));
                                        if(!fastermul_ret_v)
                                        {
                                            errs()<<"Cannot create the call inst of faster_mul func\n";
                                            exit(1);
                                        }
                                        call_inst->replaceAllUsesWith(fastermul_ret_v);

                                        inst_del.push_back(call_inst);
                                    }
                                }
                                else
                                {
                                    //NOTE.: replace original gemmex with tuning gemmex, its result is output to the optimized data file
                                    //In the last two pass(next_func_id is always -1), so it wont come down here.
                                    //std::cout<<gemm_id_tracer<<"?="<<next_func_id<<std::endl;
                                    if(gemm_id_tracer == next_func_id)
                                    {
                                        std::cout<<"The "<<gemm_id_tracer<<"th GemmEx is next to tune"<<std::endl;
                                        //TO.DO.: replace it with tuning_gemmex                                                 //DONE
                                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                                        //get all the args
                                        std::vector<Value*> args;
                                        unsigned int args_num = call_inst->arg_size();
                                        for(int i = 0; i < args_num; i++)
                                        {
                                            Value * argv = call_inst->getArgOperand(i);
                                            args.push_back(argv);
                                        }

                                        IRBuilder<> builder(dyn_cast<Instruction>(inst));
                                        ConstantInt * ID = builder.getInt32(gemm_id_tracer);
                                        Value * argv_id = dynamic_cast<Value*>(ID);

                                        char optimized_data_file_path_char[200];
                                        strcpy(optimized_data_file_path_char, optimized_data_file_path.c_str());
                                        Value * argv_optimized_data_file_path = builder.CreateGlobalStringPtr(optimized_data_file_path_char);
                                        
                                        std::string Matrix_File_Name = father_path + "GEMM_" + std::to_string(gemm_id_tracer) + ".bin";
                                        char matrix_file_path_char[200];
                                        strcpy(matrix_file_path_char,Matrix_File_Name.c_str());
                                        Value * argv_matrix_file_path = builder.CreateGlobalStringPtr(matrix_file_path_char);

                                        Value * argv_poolptr = builder.CreateLoad(gvar_ptr_mempool,"mzw_pool_ptr");
                                        if(argv_poolptr == nullptr)
                                        {
                                            errs()<<"Cannot get poolptr arg for faster_mul func\n";
                                            exit(1);
                                        }

                                        args.push_back(argv_id);
                                        args.push_back(argv_optimized_data_file_path);
                                        args.push_back(argv_matrix_file_path);
                                        args.push_back(argv_poolptr);

                                        Value * tuning_gemm_ret_v = builder.CreateCall(tuning_gemm_func_type,tuning_gemm_func_ptr,makeArrayRef(args));
                                        if(!tuning_gemm_ret_v)
                                        {
                                            errs()<<"Cannot create call inst of tuning gemm func for next target GEMM func\n";
                                            exit(1);
                                        }
                                        call_inst->replaceAllUsesWith(tuning_gemm_ret_v);

                                        inst_del.push_back(call_inst);
                                    }
                                    else
                                    {
                                        continue;
                                    }
                                }
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                }
            }
        }

        for(auto inst : inst_del)
        {
            inst->eraseFromParent();
        }
        
        return true;
    }


}

char TUNING::ID = 0;

static RegisterPass<TUNING> X("tuning","A Pass to Auto-Tuning all gemmex with low precision gemmex",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new TUNING());});