use bs58;
use generic_array::typenum::U32;
use generic_array::GenericArray;
use std::fmt;

#[derive(Serialize, Deserialize, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// define the public Pubkey tuple structure
pub struct Pubkey(GenericArray<u8, U32>);

/// implementing new  method on Pubkey structure
impl Pubkey {
    /// define new function, 
    /// and return a Pubkey structure by cloning from a slice "pubkey_vec" to construct a GenericArray  
    pub fn new(pubkey_vec: &[u8]) -> Self {
        Pubkey(GenericArray::clone_from_slice(&pubkey_vec))
    }
}

/// define as_ref function to implementing AsRef<[u8]> trait on  Pubkey structure,
/// and will return a reference to the GenericArray‘s first element of type [u8]
impl AsRef<[u8]> for Pubkey {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}

impl fmt::Debug for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

impl fmt::Display for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}
