/*
use bs58;
use generic_array::typenum::U32;
use generic_array::GenericArray;
use std::fmt;

#[derive(Serialize, Deserialize, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Pubkey(GenericArray<u8, U32>);
the source code for this can be found here https://github.com/fizyk20/generic-array/blob/30f0cc938f6cfcd41da42966f990a00653abbc7d/src/impls.rs
*/
// Pubkey is a struct returns a array of [T;N], An array of N elements of type T is written in Rust as [T; N].
// This can be treated as a array struct
package pgminterface

import (
	"fmt"
	"encoding/hex"
)

//PubkeyArray is a struct
type PubkeyArray struct {
	T	[]uint8
	N 	uint32
	Cap uint32
}

//Pubkey returns an array struct of uint8, with the size of uint32
//search ""
type Pubkey []uint8

//Array is a struct
type Array struct {
	T   []uint8 //values of the array
	N   uint32  //current length of the array
	Cap uint32  //cap length of the array, possible maximum
}

// NewArray is a method to create a new array using std lib in golang
func NewArray(inputSlice []uint8, arrayCapacity uint32) (*Array, error) {

	//error check:
	if arrayCapacity < uint32(len(inputSlice)) {
		return nil, fmt.Errorf("The array length excceds the capacity, consider reduce the size of the array")
	}

	//use make to create a new slice, each item is int8 type, 0 lengh, the cap is arrayCapacity uint32 type
	newArry := make([]uint8, 0, arrayCapacity)

	//use append to add item to the slice
	newArry = append(newArry, inputSlice[:]...)

	//use new keyword to create a new Array
	outArry := new(Array)
	outArry.T = newArry
	//use uint, len keyword
	outArry.N = uint32(len(outArry.T))
	//cap keyword
	outArry.Cap = uint32(cap(outArry.T))
	return outArry, nil
}

/*
impl Pubkey {
    pub fn new(pubkey_vec: &[u8]) -> Self {
        Pubkey(GenericArray::clone_from_slice(&pubkey_vec))
    }
}
clone_from_slice returns a generci array wiht input from pubkey_vec
*/
// New is a method of sturct Pubkey, return a Pubkey struct, refer to clone_from_slice method
//  input  :an uint8 array
//  output :an array (slice ) of
func New(pubkeyVec []uint8, arrayCapacity uint32) (*PubkeyArray, error) {
	//error check:
	if arrayCapacity < uint32(len(pubkeyVec)){
		return nil, fmt.Errorf("The array length excceds the capacity, consider reduce the size of the array")
	}
	//use make to create a new slice, each item is int8 type, 0 lengh, the cap is arrayCapacity uint32 type
	newArry := make([]uint8, 0, arrayCapacity)
	//use append to add item to the slice
	newArry = append(newArry, newArry[:]...)
	//use new keyword to create a new Array
	pubkeyArry := new(PubkeyArray)
	pubkeyArry.T = newArry
	//use uint, len keyword
	pubkeyArry.N = uint32(len(pubkeyArry.T))
	//cap keyword
	pubkeyArry.Cap = uint32(cap(pubkeyArry.T))
	return pubkeyArry, nil
}

//Asref
/* return the full range of the array
impl AsRef<[u8]> for Pubkey {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}
*/
// AsRef is a Trait implementation, ref:https://doc.rust-lang.org/reference/items/implementations.html
// https://doc.rust-lang.org/stable/book/ch10-02-traits.html Implementing a trait on a type is similar to implementing regular methods.
// the Asref can be thought as a method for Pubkey
// Notes: search ":AsRef" in MVP, no where it is used, just ignore this one
func (arguments Pubkey) Asref() []uint8{
	return arguments[0:];
}

//Debug
/*
impl fmt::Debug for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}*/
// Notes: search ":Debug" in MVP, no where it is used, just ignore this one
func (arguments Pubkey) Debug() {
	//encoding := base58.FlickrEncoding // or RippleEncoding or BitcoinEncoding
	//fmt.Println("{}", encoding.encode(arguments[0]))
	fmt.Println("{}", hex.EncodeToString(arguments))
}

//Display
/*
impl fmt::Display for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}
*/
// Notes: search ":Debug" in MVP, no where it is used, just ignore this one
func (arguments Pubkey) Display() {
	//encoding := base58.FlickrEncoding // or RippleEncoding or BitcoinEncoding
	//fmt.Println("{}", encoding.encode(arguments[0]))
	fmt.Println("{}", hex.EncodeToString(arguments))
}
