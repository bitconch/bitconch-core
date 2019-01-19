# A Simple Go And Rust Dev  (ASGARD)
Both 

## 1. Function

**Rust**


**Golang**

```go

func NewBoundContract(address common.Address, abi abi.ABI, caller ContractCaller, transactor ContractTransactor, filterer ContractFilterer) *BoundContract {
	return &BoundContract{
		address:    address,
		abi:        abi,
		caller:     caller,
		transactor: transactor,
		filterer:   filterer,
	}
}

```


## 2. Variables


| Variables        | Rust           | Go  |
| ------------- |:-------------:| -----:|
| The pointer-sized unsigned integer type.The size of this primitive is how many bytes it takes to reference any location in memory. For example, on a 32 bit target, this is 4 bytes and on a 64 bit target, this is 8 bytes.      | usize |  uintptr   |
| The 16-bit unsigned integer type.     | u16 | uint16  |
| The 32-bit unsigned integer type.      | u32 | uint32  |
| The 64-bit unsigned integer type.      | u64 | unint64  |
| A contiguous growable array type, written Vec<T> but pronounced 'vector'.      | Vec |  []type |
|       | i32 | int32  |
|       | i64 | int64  |
|       | Iterator |   |
|       | bool | bool  |



## 3.Handling Error

**Golang**

```go

// Pack the input, call and unpack the results
	input, err := c.abi.Pack(method, params...)
	if err != nil {
		return err
	}

```

**Rust**


## 4.Pointer 

**Golang**

For an operand x of type T, the address operation &x generates a pointer of type *T to x.

```go

&x
&a[f(2)]
&Point(2,3)
*p
*pf(x)


```

**Rust**

```rust



```



## 5.Define a method 

**1 -  struct**

**Golang**

```go
// BoundContract is the base wrapper object that reflects a contract on the
// Ethereum network. It contains a collection of methods that are used by the
// higher level contract bindings to operate.
type BoundContract struct {
	address    common.Address     // Deployment address of the contract on the Ethereum blockchain
	abi        abi.ABI            // Reflect based ABI to access the correct Ethereum methods
	caller     ContractCaller     // Read interface to interact with the blockchain
	transactor ContractTransactor // Write interface to interact with the blockchain
	filterer   ContractFilterer   // Event filtering to interact with the blockchain
}

```

```go

// Transact invokes the (paid) contract method with params as input values.
func (c *BoundContract) Transact(opts *TransactOpts, method string, params ...interface{}) (*types.Transaction, error) {
	// Otherwise pack up the parameters and invoke the contract
	input, err := c.abi.Pack(method, params...)
	if err != nil {
		return nil, err
	}
	return c.transact(opts, &c.address, input)
}


```



## 6. Loop


**Golang**

```go

for i, x := range indexes {
		types[i] = p.resultTypes[x]
    }
    
```



# 7. Enum

**Golang**


## 99. Package 


