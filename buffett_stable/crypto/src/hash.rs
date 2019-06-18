//! The `hash` module provides functions for creating SHA-256 hashes.

use bs58;
use generic_array::typenum::U32;
use generic_array::GenericArray;
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Serialize, Deserialize, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Hash(GenericArray<u8, U32>);

#[derive(Clone, Default)]
pub struct Hasher {
    hasher: Sha256,
}

impl Hasher {
    pub fn hash(&mut self, val: &[u8]) -> () {
        self.hasher.input(val);
    }
    pub fn hashv(&mut self, vals: &[&[u8]]) -> () {
        for val in vals {
            self.hash(val);
        }
    }
    pub fn result(self) -> Hash {
        Hash(GenericArray::clone_from_slice(
            self.hasher.result().as_slice(),
        ))
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

impl Hash {
    pub fn new(hash_slice: &[u8]) -> Self {
        Hash(GenericArray::clone_from_slice(&hash_slice))
    }
}

pub fn hashv(vals: &[&[u8]]) -> Hash {
    let mut hasher = Hasher::default();
    hasher.hashv(vals);
    hasher.result()
}


pub fn hash(val: &[u8]) -> Hash {
    hashv(&[val])
}


