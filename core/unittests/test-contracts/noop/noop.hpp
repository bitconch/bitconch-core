/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

class [[bitconchio::contract]] noop : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void anyaction( bitconchio::name                       from,
                   const bitconchio::ignore<std::string>& type,
                   const bitconchio::ignore<std::string>& data );
};
