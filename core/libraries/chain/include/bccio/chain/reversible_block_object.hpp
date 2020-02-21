
/**
 *  @file
 *  @copyright defined in bcc/LICENSE
 */
#pragma once
#include <bccio/chain/types.hpp>
#include <bccio/chain/authority.hpp>
#include <bccio/chain/block_timestamp.hpp>
#include <bccio/chain/contract_types.hpp>

#include "multi_index_includes.hpp"

namespace bccio { namespace chain {

   class reversible_block_object : public chainbase::object<reversible_block_object_type, reversible_block_object> {
      OBJECT_CTOR(reversible_block_object,(packedblock) )

      id_type        id;
      uint32_t       blocknum = 0; //< blocknum should not be changed within a chainbase modifier lambda
      shared_string  packedblock;

      void set_block( const signed_block_ptr& b ) {
         packedblock.resize( fc::raw::pack_size( *b ) );
         fc::datastream<char*> ds( packedblock.data(), packedblock.size() );
         fc::raw::pack( ds, *b );
      }

      signed_block_ptr get_block()const {
         fc::datastream<const char*> ds( packedblock.data(), packedblock.size() );
         auto result = std::make_shared<signed_block>();
         fc::raw::unpack( ds, *result );
         return result;
      }

      block_id_type get_block_id()const {
         fc::datastream<const char*> ds( packedblock.data(), packedblock.size() );
         block_header h;
         fc::raw::unpack( ds, h );
         // Only need the block id to then look up the block state in fork database, so just unpack the block_header from the stored packed data.
         // Avoid calling get_block() since that constructs a new signed_block in heap memory and unpacks the full signed_block from the stored packed data.
         return h.id();
      }
   };

   struct by_num;
   using reversible_block_index = chainbase::shared_multi_index_container<
      reversible_block_object,
      indexed_by<
         ordered_unique<tag<by_id>, member<reversible_block_object, reversible_block_object::id_type, &reversible_block_object::id>>,
         ordered_unique<tag<by_num>, member<reversible_block_object, uint32_t, &reversible_block_object::blocknum>>
      >
   >;

} } // bccio::chain

CHAINBASE_SET_INDEX_TYPE(bccio::chain::reversible_block_object, bccio::chain::reversible_block_index)
