# A Simple Guide to Port from Rust to Go

## Package Similarity

| Rust Packge         | Go Equals           | Notes  |
| ------------- |:-------------:| -----:|
|   |   |   |
| use bincode::{deserialize, serialize}; |   |   |
| use budget_instruction::Vote; |   |   |
| use choose_gossip_peer_strategy::{ChooseGossipPeerStrategy, ChooseWeightedPeerStrategy}; |   |   |
| use counter::Counter;  |   |   |
| use hash::Hash;  |   |   |
| use ledger::LedgerWindow; |   |   |
| use log::Level; |   |   |
| use netutil::{bind_in_range, bind_to, multi_bind_in_range}; |   |   |
| use packet::{to_blob, Blob, SharedBlob, BLOB_SIZE}; |   |   |
| use rand::{thread_rng, Rng}; |   |   |
| use rayon::prelude::*; |   |   |
| use result::{Error, Result}; |   |   |
| use signature::{Keypair, KeypairUtil}; |   |   |
| use solana_program_interface::pubkey::Pubkey; |   |   |
| use std; |   |   |
| use std::collections::HashMap; |   |   |
| use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket}; |   |   |
| use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering}; |   |   |
| use std::sync::{Arc, RwLock}; |   |   |
| use std::thread::{sleep, Builder, JoinHandle}; |   |   |
| use std::time::{Duration, Instant}; |   |   |
| use streamer::{BlobReceiver, BlobSender}; |   |   |
| use timing::{duration_as_ms, timestamp}; |   |   |
| use window::{SharedWindow, WindowIndex}; |   |   |