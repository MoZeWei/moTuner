# HIPCC_LLVM

## 使用前配备

* 使用下列命令配置初始环境。
```shell
spack load vim
spack load hip@4.3.1%gcc@8.5.0
spack load rocblas@4.3.1%gcc@8.5.0
spack load git
spack load gcc@8.5.0%gcc@4.8.5
spack load rocsolver@4.3.1
spack load rocminfo@4.3.1%gcc@8.5.0
spack load cmake@3.21.4%gcc@8.5.0
spack load /sinew55
```

* 将Spack中安装的HIPCC复制到自己的文件夹中，并将其路径添加到系统路径的开头。
```shell
spack find -p llvm-amdgpu@4.3.1
cp /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/llvm-amdgpu-4.3.1-mk5b5zrnseww52hd3dxklqxhv2zxphbd ~/${your_path_for_llvm}
export PATH=${your_path_for_llvm}/bin:$PATH
```

* 将复制得到的hipcc的`include`文件夹替换为本仓库的`include`文件夹

* 编写自己的rocblas_tool.hip（仓库中有）。该文件用于优化过程中的包装函数实现以及QUANTENSOR的实现。

* 将本仓库内的脚本放到任意目录`${SHELL_DIR}`下，并将其中包含`mzw`的目录替换为你自己的对应目录。

* 编写自己的rocblas_tool.hip（仓库中有）。该文件用于优化过程中的包装函数实现以及QUANTENSOR的实现。

## 使用

* 修改`${your_path_for_llvm}/include/llvm/Transform/SbSTuning/Tuning.cpp`中`TUNING()`对指定误差的赋值。
```C++
TUNING::TUNING() : ModulePass(ID)
    {
        max_err_threshold = 0.1;
        avg_err_threshold = std::numeric_limits<float>::max();
        last_tuned_id = -1;                                 //To check when first tuning
    }
```

* 进入到`${your_path_for_llvm}/include/llvm/Transform/build`并编译。将编译后得到的新动态库复制到指定地方。
```shell
make
cp SbSTuning/libTUNING.so ${your_path_for_llvm}/lib/
```

* 进入到要优化程序的代码文件夹中，将所有cpp/hip文件通过脚本转换为IR文件。
```shell
cd ${TARGET_DIR}
sh ${SHELL_DIR}/$get_ll.sh
```

* 使用llvm-link将所有IR文件链接在一起，并将得到的位码文件反汇编。
```shell
llvm-link xxx.ll xxx.ll xxx.ll -o combine.bc
llvm-dis combine.bc
```

* 对`combine.ll`使用 `whole_proc.sh`进行优化。

* 最后得到一个可执行文件，其为优化后的程序。

以上脚本均需参数输入，具体的可以看脚本内容，不复杂。

## T.I.P.S.:

* 将所有文件变为.ll文件最好使用脚本进行转换，更方便（稍后我有空的时候会放出来）
* 如有其他问题可以再问我，这里可能写得不全。
