use chrono::prelude::*;
use crate::seal::{Seal};
use buffett_interface::pubkey::Pubkey;

///This enum is defined to classify conditions by which a program
///can determine whether or not to execute a smart contract.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Condition {

    ///A condition for triggering execution of a smart contract.
    ///It requires that time hass passed the specified moment and
    ///the public key is the same as the specified one.
    Timestamp(DateTime<Utc>, Pubkey),

    ///A condition requiring only the conformation of the public
    ///key for the execution of a smart contract.
    Signature(Pubkey),
}

impl Condition {
    
    ///Check whether or not a 'Condition' variant's requirements are met by the arguments
    ///'witness' and 'from'.
    pub fn is_satisfied(&self, witness: &Seal, from: &Pubkey) -> bool {
        match (self, witness) {
            (Condition::Signature(pubkey), Seal::Signature) => pubkey == from,
            (Condition::Timestamp(dt, pubkey), Seal::Timestamp(last_time)) => {
                pubkey == from && dt <= last_time
            }
            _ => false,
    
        }
    }
}

