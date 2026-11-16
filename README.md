# BiOpaqueSymMem Opaque Predicates
Built for LLVM 19.1 - Windows x64 <br> <br>
The idea is quite simple, instead of using known values that are static on the stack, we use a table look up that is encrypted, decrypting it at runtime, this means the table values are none static. <br>
By comparing none static values in memory, we effectively break most existing symbolic execution SMT solvers, as they won't touch anything with a 10 foot pole that isn't directly on stack. The idea is to mix this in with other passes such as the bitwise/rigged horners, storing the values in the runtime table.<br>
This paper proved its capability to defeat Angr, BAP, And Triton [Manufacturing Resilient Bi-Opaque Predicates against Symbolic Execution](https://www.cse.cuhk.edu.hk/lyu/_media/conference/hxu_dsn2018_manufacturing.pdf).<br>
<br>
<img width="1975" height="919" alt="image" src="https://github.com/user-attachments/assets/2d0ca78f-515a-4667-8699-dece3f799ab7" /><br>
<img width="1992" height="947" alt="image" src="https://github.com/user-attachments/assets/ad4d36dd-b47d-4b94-a190-30053d9b136b" /><br>
<img width="1981" height="931" alt="image" src="https://github.com/user-attachments/assets/dbacb00e-7a7e-4e11-802f-4a18e0907735" /><br>
<img width="1982" height="960" alt="image" src="https://github.com/user-attachments/assets/6c3eb150-5697-405d-ae10-5bc5680e9bc5" /><br>


## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
