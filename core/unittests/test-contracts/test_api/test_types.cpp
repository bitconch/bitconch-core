/**
 *  @file
 *  @copyright defined in bitconch/LICENSE
 */
#include <bitconchiolib/bitconchio.hpp>

#include "test_api.hpp"

void test_types::types_size() {

   bitconchio_assert( sizeof(int64_t)   ==  8, "int64_t size != 8"   );
   bitconchio_assert( sizeof(uint64_t)  ==  8, "uint64_t size != 8"  );
   bitconchio_assert( sizeof(uint32_t)  ==  4, "uint32_t size != 4"  );
   bitconchio_assert( sizeof(int32_t)   ==  4, "int32_t size != 4"   );
   bitconchio_assert( sizeof(uint128_t) == 16, "uint128_t size != 16");
   bitconchio_assert( sizeof(int128_t)  == 16, "int128_t size != 16" );
   bitconchio_assert( sizeof(uint8_t)   ==  1, "uint8_t size != 1"   );

   bitconchio_assert( sizeof(bitconchio::name) ==  8, "name size !=  8");
}

void test_types::char_to_symbol() {

   bitconchio_assert( bitconchio::name::char_to_value('1') ==  1, "bitconchio::char_to_symbol('1') !=  1" );
   bitconchio_assert( bitconchio::name::char_to_value('2') ==  2, "bitconchio::char_to_symbol('2') !=  2" );
   bitconchio_assert( bitconchio::name::char_to_value('3') ==  3, "bitconchio::char_to_symbol('3') !=  3" );
   bitconchio_assert( bitconchio::name::char_to_value('4') ==  4, "bitconchio::char_to_symbol('4') !=  4" );
   bitconchio_assert( bitconchio::name::char_to_value('5') ==  5, "bitconchio::char_to_symbol('5') !=  5" );
   bitconchio_assert( bitconchio::name::char_to_value('a') ==  6, "bitconchio::char_to_symbol('a') !=  6" );
   bitconchio_assert( bitconchio::name::char_to_value('b') ==  7, "bitconchio::char_to_symbol('b') !=  7" );
   bitconchio_assert( bitconchio::name::char_to_value('c') ==  8, "bitconchio::char_to_symbol('c') !=  8" );
   bitconchio_assert( bitconchio::name::char_to_value('d') ==  9, "bitconchio::char_to_symbol('d') !=  9" );
   bitconchio_assert( bitconchio::name::char_to_value('e') == 10, "bitconchio::char_to_symbol('e') != 10" );
   bitconchio_assert( bitconchio::name::char_to_value('f') == 11, "bitconchio::char_to_symbol('f') != 11" );
   bitconchio_assert( bitconchio::name::char_to_value('g') == 12, "bitconchio::char_to_symbol('g') != 12" );
   bitconchio_assert( bitconchio::name::char_to_value('h') == 13, "bitconchio::char_to_symbol('h') != 13" );
   bitconchio_assert( bitconchio::name::char_to_value('i') == 14, "bitconchio::char_to_symbol('i') != 14" );
   bitconchio_assert( bitconchio::name::char_to_value('j') == 15, "bitconchio::char_to_symbol('j') != 15" );
   bitconchio_assert( bitconchio::name::char_to_value('k') == 16, "bitconchio::char_to_symbol('k') != 16" );
   bitconchio_assert( bitconchio::name::char_to_value('l') == 17, "bitconchio::char_to_symbol('l') != 17" );
   bitconchio_assert( bitconchio::name::char_to_value('m') == 18, "bitconchio::char_to_symbol('m') != 18" );
   bitconchio_assert( bitconchio::name::char_to_value('n') == 19, "bitconchio::char_to_symbol('n') != 19" );
   bitconchio_assert( bitconchio::name::char_to_value('o') == 20, "bitconchio::char_to_symbol('o') != 20" );
   bitconchio_assert( bitconchio::name::char_to_value('p') == 21, "bitconchio::char_to_symbol('p') != 21" );
   bitconchio_assert( bitconchio::name::char_to_value('q') == 22, "bitconchio::char_to_symbol('q') != 22" );
   bitconchio_assert( bitconchio::name::char_to_value('r') == 23, "bitconchio::char_to_symbol('r') != 23" );
   bitconchio_assert( bitconchio::name::char_to_value('s') == 24, "bitconchio::char_to_symbol('s') != 24" );
   bitconchio_assert( bitconchio::name::char_to_value('t') == 25, "bitconchio::char_to_symbol('t') != 25" );
   bitconchio_assert( bitconchio::name::char_to_value('u') == 26, "bitconchio::char_to_symbol('u') != 26" );
   bitconchio_assert( bitconchio::name::char_to_value('v') == 27, "bitconchio::char_to_symbol('v') != 27" );
   bitconchio_assert( bitconchio::name::char_to_value('w') == 28, "bitconchio::char_to_symbol('w') != 28" );
   bitconchio_assert( bitconchio::name::char_to_value('x') == 29, "bitconchio::char_to_symbol('x') != 29" );
   bitconchio_assert( bitconchio::name::char_to_value('y') == 30, "bitconchio::char_to_symbol('y') != 30" );
   bitconchio_assert( bitconchio::name::char_to_value('z') == 31, "bitconchio::char_to_symbol('z') != 31" );

   for(unsigned char i = 0; i<255; i++) {
      if( (i >= 'a' && i <= 'z') || (i >= '1' || i <= '5') ) continue;
      bitconchio_assert( bitconchio::name::char_to_value((char)i) == 0, "bitconchio::char_to_symbol() != 0" );
   }
}

void test_types::string_to_name() {
   return;
   bitconchio_assert( bitconchio::name("a") == "a"_n, "bitconchio::string_to_name(a)" );
   bitconchio_assert( bitconchio::name("ba") == "ba"_n, "bitconchio::string_to_name(ba)" );
   bitconchio_assert( bitconchio::name("cba") == "cba"_n, "bitconchio::string_to_name(cba)" );
   bitconchio_assert( bitconchio::name("dcba") == "dcba"_n, "bitconchio::string_to_name(dcba)" );
   bitconchio_assert( bitconchio::name("edcba") == "edcba"_n, "bitconchio::string_to_name(edcba)" );
   bitconchio_assert( bitconchio::name("fedcba") == "fedcba"_n, "bitconchio::string_to_name(fedcba)" );
   bitconchio_assert( bitconchio::name("gfedcba") == "gfedcba"_n, "bitconchio::string_to_name(gfedcba)" );
   bitconchio_assert( bitconchio::name("hgfedcba") == "hgfedcba"_n, "bitconchio::string_to_name(hgfedcba)" );
   bitconchio_assert( bitconchio::name("ihgfedcba") == "ihgfedcba"_n, "bitconchio::string_to_name(ihgfedcba)" );
   bitconchio_assert( bitconchio::name("jihgfedcba") == "jihgfedcba"_n, "bitconchio::string_to_name(jihgfedcba)" );
   bitconchio_assert( bitconchio::name("kjihgfedcba") == "kjihgfedcba"_n, "bitconchio::string_to_name(kjihgfedcba)" );
   bitconchio_assert( bitconchio::name("lkjihgfedcba") == "lkjihgfedcba"_n, "bitconchio::string_to_name(lkjihgfedcba)" );
   bitconchio_assert( bitconchio::name("mlkjihgfedcba") == "mlkjihgfedcba"_n, "bitconchio::string_to_name(mlkjihgfedcba)" );
}
