/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#include <bitconchio/bitconchio.hpp>

using namespace bitconchio;

extern "C" {
   void apply( uint64_t receiver, uint64_t first_receiver, uint64_t action ) {
      check( receiver == first_receiver, "rejecting all notifications" );

      // reject all actions with only the following exceptions:
      //   * do not reject an bitconchio::setcode that sets code on the bitconchio account unless the rejectall account exists;
      //   * do not reject an bitconchio::newaccount that creates the rejectall account.

      if( first_receiver == "bccio"_n.value ) {
         if( action == "setcode"_n.value ) {
            auto accnt = unpack_action_data<name>();
            if( accnt == "bccio"_n && !is_account("rejectall"_n) )
               return;
         } else if( action == "newaccount"_n.value ) {
            auto accnts = unpack_action_data< std::pair<name, name> >();
            if( accnts.second == "rejectall"_n )
               return;
         }
      }

      check( false , "rejecting all actions" );
   }
}
