/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] test_api_multi_index : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action("s1g")]]
   void idx64_general();

   [[bitconchio::action("s1store")]]
   void idx64_store_only();

   [[bitconchio::action("s1check")]]
   void idx64_check_without_storing();

   [[bitconchio::action("s1findfail1")]]
   void idx64_require_find_fail();

   [[bitconchio::action("s1findfail2")]]
   void idx64_require_find_fail_with_msg();

   [[bitconchio::action("s1findfail3")]]
   void idx64_require_find_sk_fail();

   [[bitconchio::action("s1findfail4")]]
   void idx64_require_find_sk_fail_with_msg();

   [[bitconchio::action("s1pkend")]]
   void idx64_pk_iterator_exceed_end();

   [[bitconchio::action("s1skend")]]
   void idx64_sk_iterator_exceed_end();

   [[bitconchio::action("s1pkbegin")]]
   void idx64_pk_iterator_exceed_begin();

   [[bitconchio::action("s1skbegin")]]
   void idx64_sk_iterator_exceed_begin();

   [[bitconchio::action("s1pkref")]]
   void idx64_pass_pk_ref_to_other_table();

   [[bitconchio::action("s1skref")]]
   void idx64_pass_sk_ref_to_other_table();

   [[bitconchio::action("s1pkitrto")]]
   void idx64_pass_pk_end_itr_to_iterator_to();

   [[bitconchio::action("s1pkmodify")]]
   void idx64_pass_pk_end_itr_to_modify();

   [[bitconchio::action("s1pkerase")]]
   void idx64_pass_pk_end_itr_to_erase();

   [[bitconchio::action("s1skitrto")]]
   void idx64_pass_sk_end_itr_to_iterator_to();

   [[bitconchio::action("s1skmodify")]]
   void idx64_pass_sk_end_itr_to_modify();

   [[bitconchio::action("s1skerase")]]
   void idx64_pass_sk_end_itr_to_erase();

   [[bitconchio::action("s1modpk")]]
   void idx64_modify_primary_key();

   [[bitconchio::action("s1exhaustpk")]]
   void idx64_run_out_of_avl_pk();

   [[bitconchio::action("s1skcache")]]
   void idx64_sk_cache_pk_lookup();

   [[bitconchio::action("s1pkcache")]]
   void idx64_pk_cache_sk_lookup();

   [[bitconchio::action("s2g")]]
   void idx128_general();

   [[bitconchio::action("s2store")]]
   void idx128_store_only();

   [[bitconchio::action("s2check")]]
   void idx128_check_without_storing();

   [[bitconchio::action("s2autoinc")]]
   void idx128_autoincrement_test();

   [[bitconchio::action("s2autoinc1")]]
   void idx128_autoincrement_test_part1();

   [[bitconchio::action("s2autoinc2")]]
   void idx128_autoincrement_test_part2();

   [[bitconchio::action("s3g")]]
   void idx256_general();

   [[bitconchio::action("sdg")]]
   void idx_double_general();

   [[bitconchio::action("sldg")]]
   void idx_long_double_general();

};
