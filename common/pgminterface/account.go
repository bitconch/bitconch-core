/*
use pubkey::Pubkey;
*/
package pgminterface

/*
/// An Account with userdata that is stored on chain
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Account {
    /// tokens in the account
    pub tokens: i64,
    /// user data
    /// A transaction can write to its userdata
    pub userdata: Vec<u8>,
    /// contract id this contract belongs to
    pub program_id: Pubkey,
}
*/

//Account is a data structure stores userdata in the ledger file
type Account struct {
	//
	Tokens    int64
	UserData  []uint8
	ProgramID Pubkey
}

//NewAccount is the method of Account, returns a Account struct
// search keyword：new, method , struct in go-ethereum to find similar use case
// find a similar case in go-ethereum/core/vm/logger.go
// also refer to this https://github.com/golang/go/wiki/CodeReviewComments#interfaces
func NewAccount(tokens int64, space uint64, programID Pubkey) *Account {
	newAccount := &Account{
		Tokens:    tokens,
		UserData:  make([]uint8, space),
		ProgramID: programID,
	}

	return newAccount
}

/*
input is :
            tokens i64
            space usize
            program_id Pubkey
Output Account struct with corresponding input value
 vec! is a macro returns a Vec with input arguments
impl Account {
    pub fn new(tokens: i64, space: usize, program_id: Pubkey) -> Account {
        Account {
            tokens,
            userdata: vec![0u8; space],
            program_id,
        }
    }
}

#[derive(Debug)]
pub struct KeyedAccount<'a> {
    pub key: &'a Pubkey,
    pub account: &'a mut Account,
}
*/
type KeyedAccount struct {
	Key *Pubkey
	Account Account
}
