// Copyright 2018 The bitconch-bus Authors
// This file is part of the bitconch-bus library.
//
// The bitconch-bus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The bitconch-bus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the bitconch-bus library. If not, see <http://www.gnu.org/licenses/>.

package sigverify

//create a function test_layout
func test_layout() {}

//create a function make_packet_from_transaction
func make_packet_from_transaction() {}

//create a function test_verify_n
func test_verify_n() {

}

//create a function test_verify_zero, call test_verify_n(0,false)
func test_verify_zero() {
	test_verify_n(0, false)
}

//create a function test_verify_one, call test_verify_n(1,false)
func test_verify_one() {
	test_verify_n(1, false)
}

//create a function test_verify_seventy_one, call test_verify_n(71,false)
func test_verify_zero() {
	test_verify_n(71, false)
}

//create a function ]test_verify_fail, call test_verify_n(5,true)
func test_verify_zero() {
	test_verify_n(5, true)
}
