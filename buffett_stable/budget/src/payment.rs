use buffett_interface::pubkey::Pubkey;

///This struct is defined to provide details of the tokens to be transferred
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Payment {

    ///Amount of tokens to be transferred
    pub balance: i64,

    ///Destination of the tokens to be transferred
    pub to: Pubkey,
}
