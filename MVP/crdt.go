//! The `crdt` module defines a data structure that is shared by all the nodes in the network over
//! a gossip control plane.  The goal is to share small bits of off-chain information and detect and
//! repair partitions.
//!
//! This CRDT only supports a very limited set of types.  A map of Pubkey -> Versioned Struct.
//! The last version is always picked during an update.
//!
//! The network is arranged in layers:
//!
//! * layer 0 - Leader.
//! * layer 1 - As many nodes as we can fit
//! * layer 2 - Everyone else, if layer 1 is `2^10`, layer 2 should be able to fit `2^20` number of nodes.
//!
//! Bank needs to provide an interface for us to query the stake weight

package crdt

import (
	"os"
	godebug "runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
	"net"
	"github.com/ethereum/go-ethereum/p2p/enode"
	"github.com/ethereum/go-ethereum/p2p/enr"
	"github.com/ethereum/go-ethereum/p2p/discover"
	"github.com/ethereum/go-ethereum/common/"
	"crypto/ecdsa"
	"github.com/ethereum/go-ethereum/crypto"
	"golang.org/x/crypto/ed25519/internal/edwards25519"
)

const (
	FULLNODE_PORT_RANGE = []uint16{8000, 10_000}
	
	/// milliseconds we sleep for between gossip requests
	GOSSIP_SLEEP_MILLIS = 100;
	GOSSIP_PURGE_MILLIS = 15000;

	//minimum membership table size before we start purging dead nodes
	MIN_TABLE_SIZE = 2;
)


func socketaddr(ip string, port string) {
	net.JoinHostPort(net.parseIPv4(ip), port)
}

func socketaddr_any() {
	socketaddr('0','0')
}


type CrdtError string

const (
	NoPeers CrdtError = "NoPeers"
    NoLeader CrdtError = "NoLeader"
    BadContactInfo CrdtError = "BadContactInfo"
    BadNodeInfo CrdtError= "BadNodeInfo"
    BadGossipAddress CrdtError = "BadGossipAddress"
)
	
/// Structure to be replicated by the network
type ContactInfo struct{
    /// gossip address
    ncp net.UDPAddr
    /// address to connect to for replication
    tvu net.UDPAddr
    /// address to connect to when this node is leader
    rpu net.UDPAddr
    /// transactions address
    tpu net.UDPAddr
    /// storage data address
    storage_addr net.UDPAddr
    /// if this struture changes update this value as well
    /// Always update `NodeInfo` version too
    /// This separate version for addresses allows us to use the `Vote`
    /// as means of updating the `NodeInfo` table without touching the
    /// addresses if they haven't changed.
    version int64
}

type LedgerState struct {
	last_id [32]byte
}
	
type ContactInfo struct {
    /// gossip address
    ncp string
    /// address to connect to for replication
    tvu string
    /// address to connect to when this node is leader
    rpu string
    /// transactions address
    tpu string
    /// storage data address
    storage_addr string
    /// if this struture changes update this value as well
    /// Always update `NodeInfo` version too
    /// This separate version for addresses allows us to use the `Vote`
    /// as means of updating the `NodeInfo` table without touching the
    /// addresses if they haven't changed.
    version int64
}	

type Sockets struct {
	gossip net.UDPConn
	requests net.UDPConn
	replicate []net.UDPConn
	transaction []net.UDPConn
	respond net.UDPConn
	broadcast net.UDPConn
	retransmit net.UDPConn
}

type NodeInfo struct {
	id []byte
    /// If any of the bits change, update increment this value
    version int64
    /// network addresses
    contact_info ContactInfo
    /// current leader identity
    leader_id []byte
    /// information about the state of the ledger
    ledger_state LedgerState
}

type Node struct {
	info NodeInfo
	sockets Sockets
}


func (nodeinfo *NodeInfo) new(
        id []byte,
        ncp net.UDPAddr,
        tvu net.UDPAddr,
        rpu net.UDPAddr,
        tpu net.UDPAddr,
        storage_addr net.UDPAddr,
    ) NodeInfo {
	NodeInfo{
		 id : id,
         version: 0,
         contact_info: ContactInfo {
            ncp : ncp,
            tvu : tvu,
            rpu : rpu,
            tpu : tpu,
            storage_addr : storage_addr,
            version: 0,
        },
        leader_id: common.BytesToHash('0')
        ledger_state: LedgerState {
			last_id: common.BytesToHash('0')
        },
	}
}

func (nodeinfo *NodeInfo) new_localhost(id Pubkey) NodeInfo {
	nodeinfo.new(
		id,
		socketaddr("127.0.0.1","1234"),
		socketaddr("127.0.0.1","1235"),
		socketaddr("127.0.0.1","1236"),
		socketaddr("127.0.0.1","1237"),
		socketaddr("127.0.0.1","1238"),
	)
}
// NodeInfo with unspecified addresses for adversarial testing.
func (nodeinfo *NodeInfo) new_unspecified() NodeInfo {
	addr = socketaddr("0","0")
	nodeinfo.new(,addr, addr, addr, addr, addr)
}

func (nodeinfo *NodeInfo) next_port(addr , nxt int16) (SocketAddr string){
	host, port, _ = net.SplitHostPort(addr)
	socketaddr(host,ParseInt(port) + nxt)

	
func (nodeinfo *NodeInfo) new_with_pubkey_socketaddr(pubkey []byte, bind_addr string) NodeInfo{
	transactions_addr = bind_addr
	gossip_addr = nodeinfo.next_port(bind_addr,1)
	replicate_addr = nodeinfo.next_port(bind_addr,2)
	requests_addr = nodeinfo.next_port(bind_addr,3)
	nodeinfo.new(
		pubkey,
		gossip_addr,
		replicate_addr,
		requests_addr,
		transactions_addr,
		"0.0.0.0:0",
	)
}

func (nodeinfo *NodeInfo) new_with_socketaddr(bind_addr string) NodeInfo{
	pub, _, _ := GenerateKey(rand.Reader)
	nodeinfo.new_with_pubkey_socketaddr(pub, bind_addr)
}

func (nodeinfo *NodeInfo) new_entry_point(gossip_addr string) NodeInfo{
	addr = socketaddr('0.0.0.0','0')
	pub, _, _ := GenerateKey(rand.Reader)
	nodeinfo.new(pub, *gossip_addr, daddr, daddr, daddr, daddr)
}



func (node *Node) new_localhost() Node {
	pub, _, _ := GenerateKey(rand.Reader)
	node.new_localhost_with_pubkey(pub)
}


func (node *Node) new_localhost_with_pubkey(pubkey []byte) Node {
	transaction = net.ListenUDP("udp", net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})
	gossip = net.ListenUDP("udp", net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})
	replicate = net.ListenUDP("udp", net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})
	requests = net.ListenUDP("udp", net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})
	repair = net.ListenUDP("udp", net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})

	respond = net.ListenUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: 0})
	broadcast = net.ListenUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: 0})
	retransmit = net.ListenUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: 0})
	storage = net.ListenUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: 0})
	info = NodeInfo.new(
		pubkey
		gossip
		replicate
		requests
		transaction
		storage
	)
	 Node {
            info
            sockets: Sockets {
                gossip
                requests
                replicate: vec![replicate],
                transaction: vec![transaction],
                respond,
                broadcast,
                repair,
                retransmit,
            },
        }
}

func (node *Node) new_with_external_ip(pubkey []byte], ncp &net.UDPAddr) {
	inc := func bind() -> (u16, UdpSocket) {
		bind_in_range(FULLNODE_PORT_RANGE).expect("Failed to bind")
	}
}

func bind_in_range(range (int16, int16)) (int16, net.UDPConn) {
    let sock = udp_socket(false)?;

    let (start, end) = range;
    let mut tries_left = end - start;
    loop {
        let rand_port = thread_rng().gen_range(start, end);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), rand_port);

        match sock.bind(&SockAddr::from(addr)) {
            Ok(_) => {
                let sock = sock.into_udp_socket();
                break Result::Ok((sock.local_addr().unwrap().port(), sock));
            }
            Err(err) => if err.kind() != io::ErrorKind::AddrInUse || tries_left == 0 {
                return Err(err);
            },
        }
        tries_left -= 1;
    }
}