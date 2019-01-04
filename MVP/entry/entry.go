package entry

import (
	"github.com/Bitconch/BUS/common"
	"github.com/Bitconch/BUS/MVP/transaction/Transaction"
	"github.com/Bitconch/BUS/MVP/poh/Poh"
	"github.com/Bitconch/BUS/log"
)

var EntrySender = make(chan []Entry)
var EntryReceiver = make(chan []Entry)

type Entry struct {
	NumHashes uint64
	Id common.Hash
	Transactions []Transaction
}

/// Creates the next Entry `num_hashes` after `start_hash`.
func New(start_hash common.Hash, num_hashes uint64, transactions []Transaction) *Entry {
	if transactions == nil {
		num_hashes = num_hashes + 0
	}else {
		num_hashes = num_hashes + 1
	}
	id := next_hash(start_hash, 0, transactions)
	entry := &Entry {num_hashes,id,transactions,}
	size := serialized_size(entry)
	if size > BLOB_DATA_SIZE {
		log.Info("Serialized entry size too large: s% (v% transactions):", size, len(entry.Transactions))
	}
	return entry
}

func (sharedBlob *SharedBlob) ToBlob(idx uint64, id Pubkey, addr SocketAddr) *SharedBlob {
	blob := &SharedBlob.default();
	{
		blob_w := blob.write()
		pos := {
			out := Cursor.new(blob_w.data_mut())
			serialize_into(out, sharedBlob).expect("failed to serialize output")
			out.position() as usize
		}
		blob_w.set_size(pos)
		if idx != nil {
			blob_w.set_index(idx).expect("set_index()")
		}
		if id != nil {
			blob_w.set_id(id).expect("set_id()")
		}
		if addr != nil {
			blob_w.meta.set_addr(addr)
		}
		blob_w.set_flags(0)
	}
	return blob
}

//Determines whether the length of the transaction serialized is <=BLOB_DATA_SIZE
func WillFit(transactions []Transaction) bool{
	size := serialized_size(Entry{0,Hash.default(),transactions})
	return size <= BLOB_DATA_SIZE
}

func NumWillFit(transactions []Transaction) int {
	if transactions == nil {
		return 0;
	}
	var num int = len(transactions);
	upper := len(transactions);
	lower := 1; // if one won't fit, we have a lot of TODOs
	next := len(transactions); // optimistic
	{
		log.Info("num {}, upper {} lower {} next {} transactions.len() {}",num,upper,lower,next,len(transactions))
		if WillFit(transactions[:num]) {
			next = (upper + num) / 2;
			lower = num;
			log.Info("num {} fits, maybe too well? trying {}", num, next)
			//debug!("num {} fits, maybe too well? trying {}", num, next);
		} else {
			next = (lower + num) / 2;
			upper = num;
			log.Info("num {} doesn't fit! trying {}", num, next)
			//debug!("num {} doesn't fit! trying {}", num, next);
		}
		// same as last time
		if next == num {
			log.Info("converged on num {}", num)
			//debug!("converged on num {}", num);
		}
		num = next;
	}
	return num
}

/// Creates the next Tick Entry `num_hashes` after `start_hash`.
func NewMut(start_hash Hash, num_hashes uint64, transactions[]Transaction) *Entry {
	entry := New(start_hash, num_hashes, transactions);
	start_hash = entry.Id;
	num_hashes = 0;
	if(serialized_size(entry) <= BLOB_DATA_SIZE){
		log.Info("entry<=BLOB_DATA_SIZE")
	}
	return entry
}

/// Creates a Entry from the number of hashes `num_hashes` since the previous transaction
/// and that resulting `id`.
func NewTick(num_hashes uint64, id Hash) *Entry {
	return &Entry {num_hashes,id,[]Transaction}
}

/// Verifies self.id is the result of hashing a `start_hash` `self.num_hashes` times.
/// If the transaction is not a Tick, then hash that as well.
func (entry Entry) Verify(start_hash Hash) bool {
	tx_plans_verified := false
	for _, tx := range entry.Transactions {
		if tx.verify_plan()==false {
			return tx_plans_verified
		}else {
			tx_plans_verified = true
		}
	}
	if !tx_plans_verified {
		return false;
	}
	ref_hash := nextHash(start_hash, entry.NumHashes, entry.Transactions);
	if entry.Id != ref_hash {
		log.Info("next_hash is invalid expected: {:?} actual: {:?}",entry.Id, ref_hash)
		return false
	}
	return true
}

/// Creates the hash `num_hashes` after `start_hash`. If the transaction contains
/// a signature, the final hash will be a hash of both the previous ID and
/// the signature.  If num_hashes is zero and there's no transaction data,
///  start_hash is returned.
func nextHash(start_hash common.Hash, num_hashes uint64, transactions []Transaction) common.Hash {
	if num_hashes == 0 && transactions == nil {
		return start_hash;
	}
	poh := Poh.new(start_hash);

	for i := 0; i <= num_hashes; i++ {
		poh.hash();
	}
	if transactions == nil {
		poh.tick().id
	} else {
		poh.record(Transaction.hash(transactions)).id
	}
	return poh;
}

/// Creates the next Tick or Transaction Entry `num_hashes` after `start_hash`.
func NextEntry(start_hash Hash, num_hashes uint64, transactions[] Transaction) Entry {
	if(num_hashes > 0 || transactions == nil){
		log.Info("NumHashes>0 Or Transactions is Null")
	}
	return Entry {num_hashes,nextHash(start_hash, num_hashes, transactions),transactions}
}