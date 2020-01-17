/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>
#include <vector>

class [[bitconchio::contract]] deferred_test : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void defercall( bitconchio::name payer, uint64_t sender_id, bitconchio::name contract, uint64_t payload );

   [[bitconchio::action]]
   void delayedcall( bitconchio::name payer, uint64_t sender_id, bitconchio::name contract,
                     uint64_t payload, uint32_t delay_sec, bool replace_existing );

   [[bitconchio::action]]
   void deferfunc( uint64_t payload );
   using deferfunc_action = bitconchio::action_wrapper<"deferfunc"_n, &deferred_test::deferfunc>;

   [[bitconchio::action]]
   void inlinecall( bitconchio::name contract, bitconchio::name authorizer, uint64_t payload );

   [[bitconchio::action]]
   void fail();

   [[bitconchio::on_notify("bitconchio::onerror")]]
   void on_error( uint128_t sender_id, bitconchio::ignore<std::vector<char>> sent_trx );
};
