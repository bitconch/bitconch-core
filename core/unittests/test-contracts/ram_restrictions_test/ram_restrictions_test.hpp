/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] ram_restrictions_test : public bitconchio::contract {
public:
   struct [[bitconchio::table]] data {
      uint64_t           key;
      std::vector<char>  value;

      uint64_t primary_key() const { return key; }
   };

   typedef bitconchio::multi_index<"tablea"_n, data> tablea;
   typedef bitconchio::multi_index<"tableb"_n, data> tableb;

public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void noop();

   [[bitconchio::action]]
   void setdata( uint32_t len1, uint32_t len2, bitconchio::name payer );

   [[bitconchio::action]]
   void notifysetdat( bitconchio::name acctonotify, uint32_t len1, uint32_t len2, bitconchio::name payer );

   [[bitconchio::on_notify("tester2::notifysetdat")]]
   void on_notify_setdata( bitconchio::name acctonotify, uint32_t len1, uint32_t len2, bitconchio::name payer );

   [[bitconchio::action]]
   void senddefer( uint64_t senderid, bitconchio::name payer );

   [[bitconchio::action]]
   void notifydefer( bitconchio::name acctonotify, uint64_t senderid, bitconchio::name payer );

   [[bitconchio::on_notify("tester2::notifydefer")]]
   void on_notifydefer( bitconchio::name acctonotify, uint64_t senderid, bitconchio::name payer );

};
