package transaction

import (
	"github.com/Bitconch/BUS/common"
	"github.com/Bitconch/BUS/signature"
	"github.com/ethereum/go-ethereum/log"
)

//import "github.com/docker/docker/pkg/reexec"

/// An instruction signed by a client with `Pubkey`.
type Transaction struct {
	/// A digital signature of `keys`, `program_id`, `last_id`, `fee` and `userdata`, signed by `Pubkey`.
	Signature signature.Signature

	/// The `Pubkeys` that are executing this transaction userdata.  The meaning of each key is
	/// program-specific.
	/// * keys[0] - Typically this is the `caller` public key.  `signature` is verified with keys[0].
	/// In the future which key pays the fee and which keys have signatures would be configurable.
	/// * keys[1] - Typically this is the program context or the recipient of the tokens
	Keys []Pubkey

	/// The program code that executes this transaction is identified by the program_id.
	ProgramId Pubkey

	/// The ID of a recent ledger entry.
	LastId Hash

	/// The number of tokens paid for processing and storage of this transaction.
	Fee int64

	/// Userdata to be stored in the account
	Userdata []uint8
}

/// Create a signed transaction from the given `Instruction`.
/// * `fromKeypair` - The key used to sign the transaction.  This key is stored as keys[0]
/// * `transactionKeys` - The keys for the transaction.  These are the program state
///    instances or token recipient keys.
/// * `userdata` - The input data that the program will execute with
/// * `lastId` - The PoH hash.
/// * `fee` - The transaction fee.
/*
pub fn new(
	from_keypair: &Keypair,
	transaction_keys: &[Pubkey],
	program_id: Pubkey,
	userdata: Vec<u8>,
	last_id: Hash,
	fee: i64,
) -> Self {
	let from = from_keypair.pubkey();
	let mut keys = vec![from];
	keys.extend_from_slice(transaction_keys);
	let mut tx = Transaction {
		signature: Signature::default(),
		keys,
		program_id,
		last_id,
		fee,
		userdata,
	};
	tx.sign(from_keypair);
	tx
}
 */
func (transaction *Transaction) New(fromKeypair signature.Keypair, transactionKeys Pubkey, programId Pubkey, userdata []uint8, lastId common.Hash, fee int64) *Transaction {
	from := fromKeypair.pubkey()
	keys := make([]uint8, 0, from)
	keys.ExtendFromSlice(transactionKeys);
	tx := &Transaction {
		transaction.Signature,
		keys,
		programId,
		lastId,
		fee,
		userdata,
	};
	transaction.Sign(fromKeypair)
	return tx
}

/*
pub fn get_sign_data(&self) -> Vec<u8> {
	let mut data = serialize(&(&self.keys)).expect("serialize keys");

	let program_id = serialize(&(&self.program_id)).expect("serialize program_id");
	data.extend_from_slice(&program_id);

	let last_id_data = serialize(&(&self.last_id)).expect("serialize last_id");
	data.extend_from_slice(&last_id_data);

	let fee_data = serialize(&(&self.fee)).expect("serialize last_id");
	data.extend_from_slice(&fee_data);

	let userdata = serialize(&(&self.userdata)).expect("serialize userdata");
	data.extend_from_slice(&userdata);
	data
}
 */
/// Get the transaction data to sign.
func (transaction *Transaction) GetSignData() []uint8 {
	data := serialize(&(&transaction.Keys)).expect("serialize keys")

	programId := serialize(&(&transaction.ProgramId)).expect("serialize program_id")
	data.extend_from_slice(&programId)

	lastIdData := serialize(&(&transaction.LastId)).expect("serialize last_id")
	data.extend_from_slice(&lastIdData)

	feeData := serialize(&(&transaction.Fee)).expect("serialize last_id")
	data.extend_from_slice(&feeData)

	userdata := serialize(&(&transaction.Userdata)).expect("serialize userdata")
	data.extend_from_slice(&userdata)
	return data
}

/*
pub fn sign(&mut self, keypair: &Keypair) {
	let sign_data = self.get_sign_data();
	self.signature = Signature::new(keypair.sign(&sign_data).as_ref());
}
 */
/// Sign this transaction.
func (transaction *Transaction) Sign(keypair *Keypair)  {
	signData := transaction.GetSignData()
	transaction.Signature = signature.Signature.New(keypair.Sign(&signData).AsRef());
}

/*
pub fn verify_signature(&self) -> bool {
	warn!("transaction signature verification called");
	self.signature.verify(&self.from().as_ref(), &self.GetSignData())
}
 */
/// Verify only the transaction signature.
func (transaction *Transaction) VerifySignature() bool {
	log.Info("transaction signature verification called")
	return transaction.Signature.Verify(transaction.Signature.Asref(), transaction.GetSignData())
}

/*
pub fn from(&self) -> &Pubkey {
	&self.keys[0]
}
 */
func (transaction Transaction) From() Pubkey {
	return transaction.Keys[0]
}

 /*
pub fn hash(transactions: &[Transaction]) -> Hash {
	let mut hasher = Hasher::default();
	transactions
		.iter()
		.for_each(|tx| hasher.hash(&tx.signature.as_ref()));
	hasher.result()
}
*/
// a hash of a slice of transactions only needs to hash the signatures
func Hash(transactions []Transaction) common.Hash {
	hasher := []common.Hash{}
	//hasher := Hasher.default()
	for _, tran := range transactions  {
		hasher = append(hasher, tran.Signature.Asref())
	}
	return hasher
}