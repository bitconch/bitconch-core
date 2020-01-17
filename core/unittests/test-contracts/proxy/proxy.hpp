/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>
#include <bitconchio/singleton.hpp>
#include <bitconchio/asset.hpp>

// Extacted from bccio.token contract:
namespace bitconchio {
   class [[bitconchio::contract("bccio.token")]] token : public bitconchio::contract {
   public:
      using bitconchio::contract::contract;

      [[bitconchio::action]]
      void transfer( bitconchio::name        from,
                     bitconchio::name        to,
                     bitconchio::asset       quantity,
                     const std::string& memo );
      using transfer_action = bitconchio::action_wrapper<"transfer"_n, &token::transfer>;
   };
}

// This contract:
class [[bitconchio::contract]] proxy : public bitconchio::contract {
public:
   proxy( bitconchio::name self, bitconchio::name first_receiver, bitconchio::datastream<const char*> ds );

   [[bitconchio::action]]
   void setowner( bitconchio::name owner, uint32_t delay );

   [[bitconchio::on_notify("bccio.token::transfer")]]
   void on_transfer( bitconchio::name        from,
                     bitconchio::name        to,
                     bitconchio::asset       quantity,
                     const std::string& memo );

   [[bitconchio::on_notify("bitconchio::onerror")]]
   void on_error( uint128_t sender_id, bitconchio::ignore<std::vector<char>> sent_trx );

   struct [[bitconchio::table]] config {
      bitconchio::name owner;
      uint32_t    delay   = 0;
      uint32_t    next_id = 0;

      BITCONCHLIB_SERIALIZE( config, (owner)(delay)(next_id) )
   };

   using config_singleton = bitconchio::singleton< "config"_n,  config >;

protected:
   config_singleton _config;
};
