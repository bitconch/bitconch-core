/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] test_api_db : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action("pg")]]
   void primary_i64_general();

   [[bitconchio::action("pl")]]
   void primary_i64_lowerbound();

   [[bitconchio::action("pu")]]
   void primary_i64_upperbound();

   [[bitconchio::action("s1g")]]
   void idx64_general();

   [[bitconchio::action("s1l")]]
   void idx64_lowerbound();

   [[bitconchio::action("s1u")]]
   void idx64_upperbound();

   [[bitconchio::action("tia")]]
   void test_invalid_access( bitconchio::name code, uint64_t val, uint32_t index, bool store );

   [[bitconchio::action("sdnancreate")]]
   void idx_double_nan_create_fail();

   [[bitconchio::action("sdnanmodify")]]
   void idx_double_nan_modify_fail();

   [[bitconchio::action("sdnanlookup")]]
   void idx_double_nan_lookup_fail( uint32_t lookup_type );

   [[bitconchio::action("sk32align")]]
   void misaligned_secondary_key256_tests();

};
