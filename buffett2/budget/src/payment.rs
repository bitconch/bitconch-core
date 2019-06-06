use buffett_interface::pubkey::Pubkey;


#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Payment {
    pub balance: i64,
    pub to: Pubkey,
}
