/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] restrict_action_test : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void noop( );

   [[bitconchio::action]]
   void sendinline( bitconchio::name authorizer );

   [[bitconchio::action]]
   void senddefer( bitconchio::name authorizer, uint32_t senderid );


   [[bitconchio::action]]
   void notifyinline( bitconchio::name acctonotify, bitconchio::name authorizer );

   [[bitconchio::action]]
   void notifydefer( bitconchio::name acctonotify, bitconchio::name authorizer, uint32_t senderid );

   [[bitconchio::on_notify("testacc::notifyinline")]]
   void on_notify_inline( bitconchio::name acctonotify, bitconchio::name authorizer );

   [[bitconchio::on_notify("testacc::notifydefer")]]
   void on_notify_defer( bitconchio::name acctonotify, bitconchio::name authorizer, uint32_t senderid );
};
