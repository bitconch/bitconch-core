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

import (
	"fmt"
)

//define a struct "Elems"

//call extern C code inline

//create a public function Init, for golang, function starts with capital letter
func Init() {

	ed25519_set_verbose(true)

	if !ed25519_init() {
		fmt.Errorf("Initialization Failed for ed25519_init")
	}
	ed25519_set_verbose(false)

}

//create a function verify_packet
func verify_packet() {}

//create a function verify_packet_disabled
func verify_packet_disabled() {}

//create a function batch_size
func batch_size() {}

//create a public function ed25519_verify_cpu
func Verify_Ed2551_Cpu9() {}

//create a public function ed25519_verify_disabled
func Verify_Ed25519_Disabled() {}

//create a public function ed25519_verify
func Verify_Ed25519() {}
