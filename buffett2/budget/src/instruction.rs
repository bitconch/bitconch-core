use crate::budget::Budget;
use chrono::prelude::{DateTime, Utc};


#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Contract {
    
    pub tokens: i64,
    pub budget: Budget,
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Vote {
    
    pub version: u64,
    
    pub contact_info_version: u64,
    
}


#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Instruction {
    NewContract(Contract),
    ApplyDatetime(DateTime<Utc>),
    ApplySignature,
    NewVote(Vote),
}
