/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#include "noop.hpp"

using namespace bitconchio;

void noop::anyaction( name                       from,
                      const ignore<std::string>& type,
                      const ignore<std::string>& data )
{
   require_auth( from );
}
