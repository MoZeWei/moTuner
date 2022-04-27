# moTuner: A Compiler-based Auto-tuning Approach for Mixed-precision Operator

## Related Publication:

```  ```

## Installation and Enviornment Preparation

* Tested platform

| Hardware | Software |
| ---- | ---- |
| MI100 | LLVM@13.0.0 |
| EPYC 7302 | CentOS 7.9 |
| N/A | ROCBLAS@4.3.1 | 
| N/A | HIP@4.3.1 |

* Pull this resportory into your local 

```shell
git clone https://github.com/MoZeWei/moTuner.git
```

* Prepare following software (In our situation, we use spack@0.17.0 to install these software)
```shell
spack load gcc@8.5.0
spack load hip@4.3.1
spack load rocblas@4.3.1
spack load llvm@11.0.0
```

* Add our implementation pass files into your llvm directory
```shell
cd ${your_llvm_dir}/include/llvm/Transform/
cp -r ${moTuner_dir}/include/llvm/Transform/Final_Gen ./
cp -r ${moTuner_dir}/include/llvm/Transform/Mark_Dimension_FuncID ./
cp -r ${moTuner_dir}/include/llvm/Transform/Tuning ./
```

* Modify the directory in ``` ${your_llvm_dir}/include/llvm/Transform/CMakeLists.txt ```

```shell
include_directories(${your_llvm_dir}/include)
link_directories(${your_llvm_dir})
```

* Re-compile the optimization pass shared object
```shell
rm -r build
mkdir build 
cd build 
cmake ../
make
```

* Copy your optimization pass .so into your llvm lib
```shell
cp ./Final_Gen/libGen.so ${your_llvm_dir}/lib/
cp ./SbS_Tuning/libTuning.so ${your_llvm_dir}/lib/
cp ./Mark_Dimension_FuncID/libMDFID.so ${your_llvm_dir}/lib/
```

* Write your own quantization/de-dequantization/precision refinement in a file (Sample: rocblas_tool.hip)

* Put scripts in ```${moTuner_dir}/wrap_command``` into any customized directory `${SHELL_DIR}`, and replace directories with `mzw` by your own directories.


## Usage

* Modify the error threshold in `${your_llvm_dir}/include/llvm/Transform/SbSTuning/Tuning.cpp`
```C++
TUNING::TUNING() : ModulePass(ID)
    {
        max_err_threshold = 0.1;
        avg_err_threshold = std::numeric_limits<float>::max();
        last_tuned_id = -1;                                 //To check when first tuning
    }
```

* Compile the new pass .so
```shell
cd ${your_llvm_dir}/include/llvm/Transform/build
make
cp SbSTuning/libTUNING.so ${your_llvm_dir}/lib/
```
* Convert all cpp/hip files in your project into IR file
```shell
cd ${TARGET_DIR}
sh ${SHELL_DIR}/$get_ll.sh
```

* Link all your IR files and generate linked target IR file
```shell
llvm-link xxx.ll xxx.ll xxx.ll -o combine.bc
llvm-dis combine.bc
```

* Optimize target program with ```combine.ll```
```shell
sh ${SHELL_DIR}/whole_proc.sh
```

## Stay Tuned
