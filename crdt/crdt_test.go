/* test window requests respond with the right blob, and do not overrun
    #[test]
    fn run_window_request_with_backoff() {
        let window = Arc::new(RwLock::new(default_window()));

        let mut me = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        me.leader_id = me.id;

        let mock_peer = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));

        // Simulate handling a repair request from mock_peer
        let rv =
            Crdt::run_window_request(&mock_peer, &socketaddr_any!(), &window, &mut None, &me, 0);
        assert!(rv.is_none());
        let blob = SharedBlob::default();
        let blob_size = 200;
        blob.write().unwrap().meta.size = blob_size;
        window.write().unwrap()[0].data = Some(blob);

        let num_requests: u32 = 64;
        for i in 0..num_requests {
            let shared_blob = Crdt::run_window_request(
                &mock_peer,
                &socketaddr_any!(),
                &window,
                &mut None,
                &me,
                0,
            ).unwrap();
            let blob = shared_blob.read().unwrap();
            // Test we copied the blob
            assert_eq!(blob.meta.size, blob_size);

            let id = if i == 0 || i.is_power_of_two() {
                me.id
            } else {
                mock_peer.id
            };
            assert_eq!(blob.get_id().unwrap(), id);
        }
	}
	*/

	
	/*
    /// TODO: This is obviously the wrong way to do this. Need to implement leader selection,
    /// delete this test after leader selection is correctly implemented
    #[test]
    fn test_update_leader() {
        logger::setup();
        let me = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        let leader0 = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        let leader1 = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        let mut crdt = Crdt::new(me.clone()).expect("Crdt::new");
        assert_eq!(crdt.top_leader(), None);
        crdt.set_leader(leader0.id);
        assert_eq!(crdt.top_leader().unwrap(), leader0.id);
        //add a bunch of nodes with a new leader
        for _ in 0..10 {
            let mut dum = NodeInfo::new_entry_point(&socketaddr!("127.0.0.1:1234"));
            dum.id = Keypair::new().pubkey();
            dum.leader_id = leader1.id;
            crdt.insert(&dum);
        }
        assert_eq!(crdt.top_leader().unwrap(), leader1.id);
        crdt.update_leader();
        assert_eq!(crdt.my_data().leader_id, leader0.id);
        crdt.insert(&leader1);
        crdt.update_leader();
        assert_eq!(crdt.my_data().leader_id, leader1.id);
    }

    #[test]
    fn test_valid_last_ids() {
        logger::setup();
        let mut leader0 = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.2:1234"));
        leader0.ledger_state.last_id = hash(b"0");
        let mut leader1 = NodeInfo::new_multicast();
        leader1.ledger_state.last_id = hash(b"1");
        let mut leader2 =
            NodeInfo::new_with_pubkey_socketaddr(Pubkey::default(), &socketaddr!("127.0.0.2:1234"));
        leader2.ledger_state.last_id = hash(b"2");
        // test that only valid tvu or tpu are retured as nodes
        let mut leader3 = NodeInfo::new(
            Keypair::new().pubkey(),
            socketaddr!("127.0.0.1:1234"),
            socketaddr_any!(),
            socketaddr!("127.0.0.1:1236"),
            socketaddr_any!(),
            socketaddr_any!(),
        );
        leader3.ledger_state.last_id = hash(b"3");
        let mut crdt = Crdt::new(leader0.clone()).expect("Crdt::new");
        crdt.insert(&leader1);
        crdt.insert(&leader2);
        crdt.insert(&leader3);
        assert_eq!(crdt.valid_last_ids(), vec![leader0.ledger_state.last_id]);
    }

    /// Validates the node that sent Protocol::ReceiveUpdates gets its
    /// liveness updated, but not if the node sends Protocol::ReceiveUpdates
    /// to itself.
    #[test]
    fn protocol_requestupdate_alive() {
        logger::setup();
        let window = Arc::new(RwLock::new(default_window()));

        let node = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        let node_with_same_addr = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:1234"));
        assert_ne!(node.id, node_with_same_addr.id);
        let node_with_diff_addr = NodeInfo::new_with_socketaddr(&socketaddr!("127.0.0.1:4321"));

        let crdt = Crdt::new(node.clone()).expect("Crdt::new");
        assert_eq!(crdt.alive.len(), 0);

        let obj = Arc::new(RwLock::new(crdt));

        let request = Protocol::RequestUpdates(1, node.clone());
        assert!(
            Crdt::handle_protocol(&obj, &node.contact_info.ncp, request, &window, &mut None,)
                .is_none()
        );

        let request = Protocol::RequestUpdates(1, node_with_same_addr.clone());
        assert!(
            Crdt::handle_protocol(&obj, &node.contact_info.ncp, request, &window, &mut None,)
                .is_none()
        );

        let request = Protocol::RequestUpdates(1, node_with_diff_addr.clone());
        Crdt::handle_protocol(&obj, &node.contact_info.ncp, request, &window, &mut None);

        let me = obj.write().unwrap();

        // |node| and |node_with_same_addr| are ok to me in me.alive, should not be in me.alive, but
        assert!(!me.alive.contains_key(&node.id));
        // same addr might very well happen because of NAT
        assert!(me.alive.contains_key(&node_with_same_addr.id));
        // |node_with_diff_addr| should now be.
        assert!(me.alive[&node_with_diff_addr.id] > 0);
    }

    #[test]
    fn test_is_valid_address() {
        assert!(cfg!(test));
        let bad_address_port = socketaddr!("127.0.0.1:0");
        assert!(!Crdt::is_valid_address(&bad_address_port));
        let bad_address_unspecified = socketaddr!(0, 1234);
        assert!(!Crdt::is_valid_address(&bad_address_unspecified));
        let bad_address_multicast = socketaddr!([224, 254, 0, 0], 1234);
        assert!(!Crdt::is_valid_address(&bad_address_multicast));
        let loopback = socketaddr!("127.0.0.1:1234");
        assert!(Crdt::is_valid_address(&loopback));
        //        assert!(!Crdt::is_valid_ip_internal(loopback.ip(), false));
    }

    #[test]
    fn test_default_leader() {
        logger::setup();
        let node_info = NodeInfo::new_localhost(Keypair::new().pubkey());
        let mut crdt = Crdt::new(node_info).unwrap();
        let network_entry_point = NodeInfo::new_entry_point(&socketaddr!("127.0.0.1:1239"));
        crdt.insert(&network_entry_point);
        assert!(crdt.leader_data().is_none());
    }

    #[test]
    fn new_with_external_ip_test_random() {
        let ip = Ipv4Addr::from(0);
        let node = Node::new_with_external_ip(Keypair::new().pubkey(), &socketaddr!(ip, 0));
        assert_eq!(node.sockets.gossip.local_addr().unwrap().ip(), ip);
        assert!(node.sockets.replicate.len() > 1);
        for tx_socket in node.sockets.replicate.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().ip(), ip);
        }
        assert_eq!(node.sockets.requests.local_addr().unwrap().ip(), ip);
        assert!(node.sockets.transaction.len() > 1);
        for tx_socket in node.sockets.transaction.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().ip(), ip);
        }
        assert_eq!(node.sockets.repair.local_addr().unwrap().ip(), ip);

        assert!(node.sockets.gossip.local_addr().unwrap().port() >= FULLNODE_PORT_RANGE.0);
        assert!(node.sockets.gossip.local_addr().unwrap().port() < FULLNODE_PORT_RANGE.1);
        let tx_port = node.sockets.replicate[0].local_addr().unwrap().port();
        assert!(tx_port >= FULLNODE_PORT_RANGE.0);
        assert!(tx_port < FULLNODE_PORT_RANGE.1);
        for tx_socket in node.sockets.replicate.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().port(), tx_port);
        }
        assert!(node.sockets.requests.local_addr().unwrap().port() >= FULLNODE_PORT_RANGE.0);
        assert!(node.sockets.requests.local_addr().unwrap().port() < FULLNODE_PORT_RANGE.1);
        let tx_port = node.sockets.transaction[0].local_addr().unwrap().port();
        assert!(tx_port >= FULLNODE_PORT_RANGE.0);
        assert!(tx_port < FULLNODE_PORT_RANGE.1);
        for tx_socket in node.sockets.transaction.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().port(), tx_port);
        }
        assert!(node.sockets.repair.local_addr().unwrap().port() >= FULLNODE_PORT_RANGE.0);
        assert!(node.sockets.repair.local_addr().unwrap().port() < FULLNODE_PORT_RANGE.1);
    }

    #[test]
    fn new_with_external_ip_test_gossip() {
        let ip = IpAddr::V4(Ipv4Addr::from(0));
        let node = Node::new_with_external_ip(Keypair::new().pubkey(), &socketaddr!(0, 8050));
        assert_eq!(node.sockets.gossip.local_addr().unwrap().ip(), ip);
        assert!(node.sockets.replicate.len() > 1);
        for tx_socket in node.sockets.replicate.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().ip(), ip);
        }
        assert_eq!(node.sockets.requests.local_addr().unwrap().ip(), ip);
        assert!(node.sockets.transaction.len() > 1);
        for tx_socket in node.sockets.transaction.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().ip(), ip);
        }
        assert_eq!(node.sockets.repair.local_addr().unwrap().ip(), ip);

        assert_eq!(node.sockets.gossip.local_addr().unwrap().port(), 8050);
        let tx_port = node.sockets.replicate[0].local_addr().unwrap().port();
        assert!(tx_port >= FULLNODE_PORT_RANGE.0);
        assert!(tx_port < FULLNODE_PORT_RANGE.1);
        for tx_socket in node.sockets.replicate.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().port(), tx_port);
        }
        assert!(node.sockets.requests.local_addr().unwrap().port() >= FULLNODE_PORT_RANGE.0);
        assert!(node.sockets.requests.local_addr().unwrap().port() < FULLNODE_PORT_RANGE.1);
        let tx_port = node.sockets.transaction[0].local_addr().unwrap().port();
        assert!(tx_port >= FULLNODE_PORT_RANGE.0);
        assert!(tx_port < FULLNODE_PORT_RANGE.1);
        for tx_socket in node.sockets.transaction.iter() {
            assert_eq!(tx_socket.local_addr().unwrap().port(), tx_port);
        }
        assert!(node.sockets.repair.local_addr().unwrap().port() >= FULLNODE_PORT_RANGE.0);
        assert!(node.sockets.repair.local_addr().unwrap().port() < FULLNODE_PORT_RANGE.1);
    }
}
*/