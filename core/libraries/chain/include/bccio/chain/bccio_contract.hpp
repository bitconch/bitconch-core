/**
 *  @file
 *  @copyright defined in bcc/LICENSE
 */
#pragma once

#include <bccio/chain/types.hpp>
#include <bccio/chain/contract_types.hpp>

namespace bccio { namespace chain {

   class apply_context;

   /**
    * @defgroup native_action_handlers Native Action Handlers
    */
   ///@{
   void apply_bccio_newaccount(apply_context&);
   void apply_bccio_updateauth(apply_context&);
   void apply_bccio_deleteauth(apply_context&);
   void apply_bccio_linkauth(apply_context&);
   void apply_bccio_unlinkauth(apply_context&);

   /*
   void apply_bccio_postrecovery(apply_context&);
   void apply_bccio_passrecovery(apply_context&);
   void apply_bccio_vetorecovery(apply_context&);
   */

   void apply_bccio_setcode(apply_context&);
   void apply_bccio_setabi(apply_context&);

   void apply_bccio_canceldelay(apply_context&);
   ///@}  end action handlers

} } /// namespace bccio::chain
