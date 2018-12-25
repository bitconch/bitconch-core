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


// The `timing` module provides std::time utility functions.
/*use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn duration_as_us(d: &Duration) -> u64 {
    (d.as_secs() * 1000 * 1000) + (u64::from(d.subsec_nanos()) / 1_000)
}

pub fn duration_as_ms(d: &Duration) -> u64 {
    (d.as_secs() * 1000) + (u64::from(d.subsec_nanos()) / 1_000_000)
}

pub fn duration_as_s(d: &Duration) -> f32 {
    d.as_secs() as f32 + (d.subsec_nanos() as f32 / 1_000_000_000.0)
}

pub fn timestamp() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("create timestamp in timing");
    duration_as_ms(&now)
}
*/
package timer

import (
	"time"
)

type elapsedtime time.Duration

// DurationAsUs returns the duration period in microseconds
func DurationAsUs(t elapsedtime) uint64 {
	return uint64(t / time.Millisecond)
}
 

// DurationAsMs returns the duration period in milliseconds
func DurationAsMs(t time.Duration) uint64 {
	return uint64(t / time.Millisecond)
}

// DurationAsSecond returns the durtion period in seconds
func DurationAsSecond(t elapsedtime) uint64 {
	return uint64(t / time.Millisecond)
}
 

// TimeStamp converts a time.Time to a number of milliseconds since 1970.
func TimeStamp() uint64 {
    t.time

}

// Millis converts a time.Time to a number of milliseconds since 1970.
func Millis(t time.Time) uint64 {
	return uint64(t.UnixNano()) / uint64(time.Millisecond)
}

// DurationMillis converts a time.Duration to a number of milliseconds.
func DurationMillis(d time.Duration) uint64 {
	return uint64(d / time.Millisecond)
}

// MillisDuration coverts milliseconds to a time.Duration.
func MillisDuration(m uint64) time.Duration {
	return time.Duration(m) * time.Millisecond
}
