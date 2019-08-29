use chrono::prelude::*;

///This enum is a redefinition of the enum 'Condition' except that it excludes the 'Pubkey' field.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Seal {
    Timestamp(DateTime<Utc>),
    Signature,
}
