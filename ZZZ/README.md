# rustelo
Call Rust from Go, C, Python, Ruby and everything.

- File structure
```
  |-----includes               <-- C header files, upon each build will be synced from vendor/rustelo_eay/include 
  |-----gobin                 <-- The binary executable 
     |-----vendor
        |-----bus-rust         <-- The Bitconch chain lib in rust, a submodule @caesarchad/bus-rust
        |-----rustelo-easy     <-- Experimental sample, to call rust from go, a submodule @caesarchad/rustelo-easy
           |-----include       <-- The corresponding header files which will be copied to rustelo folder
           |-----src           <-- Source files in rust    
           |-----target        <-- Build output.
  |-----build.py               <-- Pyton script to build
  |-----go.mod                 <-- Defines the dependent go package
  |-----go.sum 

```