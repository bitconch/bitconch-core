//! The `hash` module provides functions for creating SHA-256 hashes.

use bs58;
use generic_array::typenum::U32;
use generic_array::GenericArray;
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Serialize, Deserialize, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// define the public Hash tuple structure
pub struct Hash(GenericArray<u8, U32>);

#[derive(Clone, Default)]
/// definition public structure Hasher
pub struct Hasher {
    hasher: Sha256,
}

/// implementing hash, hashv and result methods on Hasher structure
impl Hasher {
    /// defines the hash function and the type of return value is tuple
    pub fn hash(&mut self, val: &[u8]) -> () {
        /// digest input data of val
        self.hasher.input(val);
    }
    /// defines the hashv function and the type of return value is tuple
    pub fn hashv(&mut self, vals: &[&[u8]]) -> () {
        /// looping vals array
        for val in vals {
            /// call the function of hash
            self.hash(val);
        }
    }
    /// defines the hashv function and the type of return value is Hash structure
    pub fn result(self) -> Hash {
        /// construct a GenericArray from a slice by cloning its content
        Hash(GenericArray::clone_from_slice(
            /// read hash digest and consume hasher, and extracts a slice containing the entire array
            self.hasher.result().as_slice(),
        ))
    }
}

/// define as_ref function to implementing AsRef trait on Hash structure,
/// and will return a reference to the GenericArray‘s first element of type [u8]
impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}

/// define fmt function to implementing fmt::Debug trait on Hash structure,
/// encoding  GenericArray‘s first element into Base58 encoded strings,
/// and write strictly into the supplied output stream 'f'
/// returns 'fmt::Result' which indicates whether the operation succeeded or failed.
impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

/// define fmt function to implementing fmt::Display trait on Hash structure,
/// encoding GenericArray‘s first element into Base58 encoded strings,
/// and write strictly into the supplied output stream 'f'
/// returns 'fmt::Result' which indicates whether the operation succeeded or failed.
impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

/// defining an new method on the Hash structure
/// construct a GenericArray from a slice by cloning its content by reference parameter hash_slice
/// and return a Hash structure
impl Hash {
    pub fn new(hash_slice: &[u8]) -> Self {
        Hash(GenericArray::clone_from_slice(&hash_slice))
    }
}

/// define a publick function of hashv, and the type of its return value is Hash structure
pub fn hashv(vals: &[&[u8]]) -> Hash {
    /// returns Hasher's default value
    let mut hasher = Hasher::default();
    /// digest input data of vals
    hasher.hashv(vals);
    /// read hash digest and consume hasher
    hasher.result()
}


/// define the function of hash,
/// call the function og hashv and return value is Hash 
pub fn hash(val: &[u8]) -> Hash {
    hashv(&[val])
}


