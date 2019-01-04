package signature

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/Bitconch/BUS/crypto"
	"encoding/hex"
)

//SignatureArray is a struct
type SignatureArray struct {
	T	[]uint8
	N 	uint32
	Cap uint32
}

//Signature returns an array struct of uint8, with the size of uint32
//search ""
type Signature []uint8

/*
	pub fn new(signature_slice: &[u8]) -> Self {
        Signature(GenericArray::clone_from_slice(&signature_slice))
    }
	clone_from_slice returns a generci array wiht input from signature_slice
 */
// New is a method of sturct Signature, return a Signature struct, refer to clone_from_slice method
//  input  :an uint8 array
//  output :an array (slice ) of
func (signature *Signature) New(signatureSlice []uint8, arrayCapacity uint32) (*SignatureArray, error) {
	//error check:
	if arrayCapacity < uint32(len(signatureSlice)){
		return nil, fmt.Errorf("The array length excceds the capacity, consider reduce the size of the array")
	}
	//use make to create a new slice, each item is int8 type, 0 lengh, the cap is arrayCapacity uint32 type
	newArray := make([]uint8, 0, arrayCapacity)
	//use append to add item to the slice
	newArray = append(newArray, newArray[:]...)
	//use new keyword to create a new Array
	signatureArray := new(SignatureArray)
	signatureArray.T = newArray
	//use uint, len keyword
	signatureArray.N = uint32(len(signatureArray.T))
	//cap keyword
	signatureArray.Cap = uint32(cap(signatureArray.T))
	return signatureArray, nil
}

/*
	pub fn verify(&self, pubkey_bytes: &[u8], message_bytes: &[u8]) -> bool {
        let pubkey = Input::from(pubkey_bytes);
        let message = Input::from(message_bytes);
        let signature = Input::from(self.0.as_slice());
        signature::verify(&signature::ED25519, pubkey, message, signature).is_ok()
    }
 */
/// Verify the signature `signature` of message `msg` with the public key
/// `public_key`.
///  input  :an uint8 array
///  output :an array (slice ) of
func (arguments Signature) Verify(pubkeyBytes []uint8, messageBytes []uint8) (bool, error) {
	pubkey := Input.from(pubkeyBytes)
	message := Input.from(messageBytes)
	signature := Input.from(arguments[0])
	verifyFlag := signature.verify(crypto.Ed25519(), pubkey, message, signature).is_ok()
	return verifyFlag,nil
}

//Asref
/* return the full range of the array
impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}
*/
// AsRef is a Trait implementation, ref:https://doc.rust-lang.org/reference/items/implementations.html
// https://doc.rust-lang.org/stable/book/ch10-02-traits.html Implementing a trait on a type is similar to implementing regular methods.
// the Asref can be thought as a method for Signature
// Notes: search ":AsRef" in MVP, no where it is used, just ignore this one
func (arguments Signature) Asref() []uint8{
	return arguments[0:];
}

//Debug
/*
impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}
*/
// Notes: search ":Debug" in MVP, no where it is used, just ignore this one
func (arguments Signature) Debug() {
	//encoding := base58.FlickrEncoding // or RippleEncoding or BitcoinEncoding
	//fmt.Println("{}", encoding.encode(arguments[0]))
	fmt.Println("{}", hex.EncodeToString(arguments))
}

//Display
/*
impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", bs58::encode(self.0).into_string())
    }
}
*/
// Notes: search ":Debug" in MVP, no where it is used, just ignore this one
func (arguments Signature) Display() {
	//encoding := base58.FlickrEncoding // or RippleEncoding or BitcoinEncoding
	//fmt.Println("{}", encoding.encode(arguments[0]))
	fmt.Println("{}", hex.EncodeToString(arguments))
}

/*
pub trait KeypairUtil {
    fn new() -> Self;
    fn pubkey(&self) -> Pubkey;
}
 */
type KeypairUtil interface {
	new() KeypairUtil
	pubkey(keypairUtil KeypairUtil) KeypairUtil
}

/*
fn new() -> Self {
	let rng = rand::SystemRandom::new();
	let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).expect("generate_pkcs8");
	Ed25519KeyPair::from_pkcs8(Input::from(&pkcs8_bytes)).expect("from_pcks8")
}
 */
/// Return a new ED25519 keypair
func (keyPair *KeyPair) new() KeypairUtil {
	rng := rand.New(rand.NewSource(time.Now().Unix()));
	//pkcs8_bytes := Ed25519KeyPair::generate_pkcs8(&rng).expect("generate_pkcs8");
	pkcs8Bytes := crypto.Ed25519.GenerateKey(rng)
	return keyPair.FromPkcs8(Input.from(&pkcs8Bytes))
}

/*
fn pubkey(&self) -> Pubkey {
	Pubkey::new(self.public_key_bytes())
}
 */
/// Return the public key for the given keypair
func (keyPair *KeyPair) pubkey(keypairUtil KeypairUtil) PubKey {
	return New(keypairUtil.PublicKeyBytes())
}

//GenKeysArray is a struct
type GenKeysArray struct {
	T	[]uint8
	N 	uint32
	Cap uint32
}

/*
pub struct GenKeys {
    generator: ChaChaRng,
}
 */
//GenKeys is a struct
type GenKeys struct {
	generator ChaChaRng
}

/*
pub fn new(seed: [u8; 32]) -> GenKeys {
	let generator = ChaChaRng::from_seed(seed);
	GenKeys { generator }
}
 */
// New is a method of sturct GenKeys new, return a GenKeys struct
//  input  :an uint8
//  output :an GenKeys
func (genKeys *GenKeys) New(seed uint8) GenKeys {
	generator := ChaChaRng.FromSeed(seed)
	return GenKeys{generator}
}

/*
fn gen_seed(&mut self) -> [u8; 32] {
	let mut seed = [0u8; 32];
	self.generator.fill(&mut seed);
	seed
}
 */
// New is a method of sturct genSeed, return a 32-bit array of uint8
//  output :an 32-bit array of uint8
func (genKeys *GenKeys) genSeed() [32]uint8 {
	seed := ([32]uint8)
	genKeys.generator.fill(seed)
	return seed
}

/*
fn gen_n_seeds(&mut self, n: i64) -> Vec<[u8; 32]> {
	(0..n).map(|_| self.gen_seed()).collect()
}
 */
// New is a method of sturct GenKeysArray, return a GenKeys struct
//  input  :an int64
//  output :an array (slice ) of
func (genKeys *GenKeys) genNSeeds(n int64) (*GenKeysArray, error) {
	newArry := make([]uint8, 0, 32)
	for i:=0; i<n; i++ {
		//newArry = append(newArry, genKeys.genSeed()[:]...)
		newArry = append(newArry, genKeys.genSeed())
	}
	//use new keyword to create a new Array
	genKeysArry := new(GenKeysArray)
	genKeysArry.T = newArry
	//use uint, len keyword
	genKeysArry.N = uint32(len(genKeysArry.T))
	//cap keyword
	genKeysArry.Cap = uint32(cap(genKeysArry.T))
	return genKeysArry, nil
}

/*
pub fn gen_n_keypairs(&mut self, n: i64) -> Vec<Keypair> {
   self.gen_n_seeds(n)
	   .into_par_iter()
	   .map(|seed| Keypair::from_seed_unchecked(Input::from(&seed)).unwrap())
	   .collect()
}
*/
func (genKeys *GenKeys) GenNKeypairs(n int64) []Keypair {
	kps := make([]Keypair)
	for _, seed := range genKeys.genNSeeds(n) {
		kps = append(kps, Keypair.FromSeedUnchecked(Input.from(&seed)))
	}
	return kps
}


/*
pub fn read_pkcs8(path: &str) -> Result<Vec<u8>, Box<error::Error>> {
    let file = File::open(path.to_string())?;
    let pkcs8: Vec<u8> = serde_json::from_reader(file)?;
    Ok(pkcs8)
}
 */
func ReadPkcs8(path &str) ([]uint8, error) {
	file := File.open(path.to_string());
	pkcs8 := serde_json.from_reader(file);
	return pkcs8, nil
}

/*
pub fn read_keypair(path: &str) -> Result<Keypair, Box<error::Error>> {
    let pkcs8 = read_pkcs8(path)?;
    let keypair = Ed25519KeyPair::from_pkcs8(Input::from(&pkcs8))?;
    Ok(keypair)
}
 */
func ReadKeypair(path &str) ([]Keypair, error) {
	pkcs8 := ReadPkcs8(path)
	keypair := crypto.Ed25519(Input.from(&pkcs8))
	return keypair, nil
}