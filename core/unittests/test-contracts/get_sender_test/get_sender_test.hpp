/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#pragma once

#include <bitconchio/bitconchio.hpp>

namespace bitconchio {
   namespace internal_use_do_not_use {
      extern "C" {
         __attribute__((bitconchio_wasm_import))
         uint64_t get_sender();
      }
   }
}

namespace bitconchio {
   name get_sender() {
      return name( internal_use_do_not_use::get_sender() );
   }
}

class [[bitconchio::contract]] get_sender_test : public bitconchio::contract {
public:
   using bitconchio::contract::contract;

   [[bitconchio::action]]
   void assertsender( bitconchio::name expected_sender );
   using assertsender_action = bitconchio::action_wrapper<"assertsender"_n, &get_sender_test::assertsender>;

   [[bitconchio::action]]
   void sendinline( bitconchio::name to, bitconchio::name expected_sender );

   [[bitconchio::action]]
   void notify( bitconchio::name to, bitconchio::name expected_sender, bool send_inline );

   // bitconchio.cdt 1.6.1 has a problem with "*::notify" so hardcode to tester1 for now.
   // TODO: Change it back to "*::notify" when the bug is fixed in bitconchio.cdt.
   [[bitconchio::on_notify("tester1::notify")]]
   void on_notify( bitconchio::name to, bitconchio::name expected_sender, bool send_inline );

};
