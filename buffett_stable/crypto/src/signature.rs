use bs58;
use generic_array::typenum::U64;
use generic_array::GenericArray;
use rand::{ChaChaRng,Rng, SeedableRng};
use rayon::prelude::*;
use ring::signature::Ed25519KeyPair;
//use ring::{rand, signature};
use serde_json;
use buffett_interface::pubkey::Pubkey;
use std::error;
use std::fmt;
use std::fs::File;
use untrusted::Input;

/// "Keypair" is a new name for "Ed25519KeyPair"
pub type Keypair = Ed25519KeyPair;

#[derive(Serialize, Deserialize, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// define the public Signature tuple structure
pub struct Signature(GenericArray<u8, U64>);

/// implementing new and verify methods on Signature structure
impl Signature {
    /// define new function, and return value is Signature structure
    pub fn new(signature_slice: &[u8]) -> Self {
        /// return instantiated Signature structure by cloning the content 
        /// of reference signature_slice to construct a GenericArray 
        Signature(GenericArray::clone_from_slice(&signature_slice))
    }
    /// define verify function, and the type of return value is bool
    pub fn verify(&self, pubkey_bytes: &[u8], message_bytes: &[u8]) -> bool {
        /// construct a new Input for the input bytes of pubkey_bytes
        let pubkey = Input::from(pubkey_bytes);
        /// construct a new Input for the input bytes of message_bytes
        let message = Input::from(message_bytes);
        /// construct a new Input for input bytes the slice of the first element of the GenericArray
        let signature = Input::from(self.0.as_slice());
        /// verify the signature， and return ture
        ring::signature::verify(&ring::signature::ED25519, pubkey, message, signature).is_ok()
    }
}

impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}

pub trait KeypairUtil {
    fn new() -> Self;
    fn pubkey(&self) -> Pubkey;
}

impl KeypairUtil for Ed25519KeyPair {
    fn new() -> Self {
        let rng = ring::rand::SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).expect("generate_pkcs8");
        Ed25519KeyPair::from_pkcs8(Input::from(&pkcs8_bytes)).expect("from_pcks8")
    }

    fn pubkey(&self) -> Pubkey {
        Pubkey::new(self.public_key_bytes())
    }
}

pub struct GenKeys {
    generator: ChaChaRng,
}

impl GenKeys {
    pub fn new(seed: [u8; 32]) -> GenKeys {
        let generator = ChaChaRng::from_seed(seed);
        GenKeys { generator }
    }

    fn gen_seed(&mut self) -> [u8; 32] {
        let mut seed = [0u8; 32];
        self.generator.fill(&mut seed);
        seed
    }

    fn gen_n_seeds(&mut self, n: i64) -> Vec<[u8; 32]> {
        (0..n).map(|_| self.gen_seed()).collect()
    }

    pub fn gen_n_keypairs(&mut self, n: i64) -> Vec<Keypair> {
        self.gen_n_seeds(n)
            .into_par_iter()
            .map(|seed| Keypair::from_seed_unchecked(Input::from(&seed)).unwrap())
            .collect()
    }
}

pub fn read_pkcs8(path: &str) -> Result<Vec<u8>, Box<error::Error>> {
    let file = File::open(path.to_string())?;
    let pkcs8: Vec<u8> = serde_json::from_reader(file)?;
    Ok(pkcs8)
}

pub fn read_keypair(path: &str) -> Result<Keypair, Box<error::Error>> {
    let pkcs8 = read_pkcs8(path)?;
    let keypair = Ed25519KeyPair::from_pkcs8(Input::from(&pkcs8))?;
    Ok(keypair)
}

