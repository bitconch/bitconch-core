# A Sample to Call Rust from Go (using dynamic lib)
---

## Test

1 - Run `make build`
2 - Run `./main`
3 - "Hello Dummy" is displayed to "stdout"

Run `make build` and then `./main` to see `Rust` + `Golang` in action. You
should see `Hello John Smith!` printed to `stdout`.

## Structure

```
|___include -----------C header files
|
|___lib     -----------generated artifact file (.a for static, .so for dynamic)
|
|___rustcode-----------rust libraries
       |
       |___hello-------a rust crate






```



Then, you need to create a C header file for your library. Just copy the `libc`
types that you used.

All that is left to do is to add some `cgo`-specific comments to your Golang
code. These comments tell `cgo` where to find the library and its headers.

```go
/*
#cgo LDFLAGS: -L./lib -lhello
#include "./lib/hello.h"
*/
import "C"
```

> There should not be a newline between `*/` and `import "C"`.

A simple `make build` (use the [Makefile](Makefile) in this repository) will
result in a binary that loads your dynamic library.

# References
[`cgo`](https://blog.golang.org/c-go-cgo)  
[Rust's FFI capabilities](https://doc.rust-lang.org/book/ffi.html)
[Creating a Rust dynamic library by Andrew Oppenlander](http://oppenlander.me/articles/rust-ffi).