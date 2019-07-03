use chrono::prelude::*;
use crate::seal::{Seal};
use buffett_interface::pubkey::Pubkey;



#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Condition {
    
    Timestamp(DateTime<Utc>, Pubkey),

    
    Signature(Pubkey),
}

impl Condition {
    
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
