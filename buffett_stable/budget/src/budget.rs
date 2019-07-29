use crate::seal::{Seal};
use crate::payment::{Payment};
use crate::condition::Condition;
use buffett_interface::pubkey::Pubkey;
use std::mem;




#[repr(C)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
/// define a public enueration of Budget,its variants are Pay, After, Or, And
pub enum Budget {
    
    Pay(Payment),

    
    After(Condition, Payment),

    
    Or((Condition, Payment), (Condition, Payment)),

    
    And(Condition, Condition, Payment),
}

/// implement Budget
impl Budget {
    
    
/// define a public final_payment method on the Budget enum
/// with the parameter Budget's variants and the type of return value is Option<Payment>
    pub fn final_payment(&self) -> Option<Payment> {
/// match with Budget
        match self {
/// if Budget variants is Pay's payment, execute the branch and clone payment return it 
            Budget::Pay(payment) => Some(payment.clone()),
/// else return None
            _ => None,
        }
    }

    
/// define a public verify method on the Budget enum
/// with the parameter of Budget's filed and spendable_balance, the type of return value is bool
    pub fn verify(&self, spendable_balance: i64) -> bool {
/// match with Budget
        match self {
/// if Budget filed is Pay's payment, or After'payment, or And'payment
/// execute the branch and return true or false
            Budget::Pay(payment) | Budget::After(_, payment) | Budget::And(_, _, payment) => {
                payment.balance == spendable_balance
            }
/// if Budget filed is Or((Condition, Payment), (Condition, Payment))
/// then execute this branch and returns true or flase
            Budget::Or(a, b) => a.1.balance == spendable_balance && b.1.balance == spendable_balance,
        }
    }

    
/// define a public apply_seal method on the Budget enum
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

