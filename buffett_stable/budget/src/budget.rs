use crate::seal::{Seal};
use crate::payment::{Payment};
use crate::condition::Condition;
use buffett_interface::pubkey::Pubkey;
use std::mem;

///This enum is defined to set a scheme for classification of smart contracts
#[repr(C)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum Budget {

    ///State indicating that all requirements for execution of
    ///a smart contract have been satisfied.
    Pay(Payment),

    ///State indicating that only one requirement is to be satisfied
    ///to trigger the execution of a smart contract.
    After(Condition, Payment),

    ///State indicating that either requirement is to be satisfied
    ///to trigger the execution of a smart contract.
    Or((Condition, Payment), (Condition, Payment)),

    ///State indicating that both requirements are to be satisfied
    ///to trigger the execution of a smart contract.
    And(Condition, Condition, Payment),
}

impl Budget {
    
    ///Check whether or not a Budget variant is a 'Pay(payment)', return an 
    ///Option variant 'Some' wrapping the payment if so, or an Option variant
    ///'None' if not.
    pub fn final_payment(&self) -> Option<Payment> {
        match self {
            Budget::Pay(payment) => Some(payment.clone()),
            _ => None,
        }
    }

    ///Check whether or not the balance contained in the 'payment' field of a Budget variant
    ///is equal to the argument 'spendable_balance', return the bool value of the comparison.
    pub fn verify(&self, spendable_balance: i64) -> bool {
        match self {
            Budget::Pay(payment) | Budget::After(_, payment) | Budget::And(_, _, payment) => {
                payment.balance == spendable_balance
            }
            Budget::Or(a, b) => a.1.balance == spendable_balance && b.1.balance == spendable_balance,
        }
    }

    ///To change a budget variant to another according to what kind of variant it is now and
    ///whether or not the requirements in its 'condition' field are met by the arguments
    ///'seal and 'from'.
    pub fn apply_seal(&mut self, seal: &Seal, from: &Pubkey) {
        let new_budget = match self {
            Budget::After(condition, payment) if condition.is_satisfied(seal, from) => {
                Some(Budget::Pay(payment.clone()))
            }
            Budget::Or((condition, payment), _) if condition.is_satisfied(seal, from) => {
                Some(Budget::Pay(payment.clone()))
            }
            Budget::Or(_, (condition, payment)) if condition.is_satisfied(seal, from) => {
                Some(Budget::Pay(payment.clone()))
            }
            Budget::And(condition1, condition2, payment) => {
                if condition1.is_satisfied(seal, from) {
                    Some(Budget::After(condition2.clone(), payment.clone()))
                } else if condition2.is_satisfied(seal, from) {
                    Some(Budget::After(condition1.clone(), payment.clone()))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(budget) = new_budget {
            mem::replace(self, budget);
        }
    }
}

