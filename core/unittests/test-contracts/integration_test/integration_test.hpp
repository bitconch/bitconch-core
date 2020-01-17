/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] integration_test : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void store( bitconchio::name from, bitconchio::name to, uint64_t num );

   struct [[bitconchio::table("payloads")]] payload {
      uint64_t              key;
      std::vector<uint64_t> data;

      uint64_t primary_key()const { return key; }

      BITCONCHLIB_SERIALIZE( payload, (key)(data) )
   };

   using payloads_table = bitconchio::multi_index< "payloads"_n,  payload >;

};
