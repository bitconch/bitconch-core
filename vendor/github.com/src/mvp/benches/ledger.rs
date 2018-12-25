#![feature(test)]
extern crate buffett;
extern crate test;

use buffett::hash::{hash, Hash};
use buffett::ledger::{next_entries, reconstruct_entries_from_blobs, Block};
use buffett::signature::{Keypair, KeypairUtil};
use buffett::system_transaction::SystemTransaction;
use buffett::transaction::Transaction;
use test::Bencher;

#[bench]
fn bench_block_to_blobs_to_block(bencher: &mut Bencher) {
    let zero = Hash::default();
    let one = hash(&zero.as_ref());
    let keypair = Keypair::new();
    let tx0 = Transaction::system_move(&keypair, keypair.pubkey(), 1, one, 0);
    let transactions = vec![tx0; 10];
    let entries = next_entries(&zero, 1, transactions);

    bencher.iter(|| {
        let blobs = entries.to_blobs();
        assert_eq!(reconstruct_entries_from_blobs(blobs).unwrap(), entries);
    });
}
