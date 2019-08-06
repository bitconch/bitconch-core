use buffett_interface::pubkey::Pubkey;

///This struct is defined to collect information on the recipient of a payment.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Payment {

    ///Balance of the recipient's account.
    pub balance: i64,

    ///The public key of the recipient.
    pub to: Pubkey,
}

