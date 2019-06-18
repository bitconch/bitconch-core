use chrono::prelude::*;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Seal {
    Timestamp(DateTime<Utc>),
    Signature,
}
