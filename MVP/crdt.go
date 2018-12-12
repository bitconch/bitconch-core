import (
	"fmt"
	"math"
	"os"
	godebug "runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
	"net"
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
	
type LedgerState struct {
	last_id []int8
}
	
pub struct ContactInfo {
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
	
type NodeInfo struct {
	id []byte
    /// If any of the bits change, update increment this value
    version int64
    /// network addresses
    pub contact_info ContactInfo
    /// current leader identity
    pub leader_id []byte
    /// information about the state of the ledger
    pub ledger_state LedgerState
}


func (nodeinfo *NodeInfo) new(
        id []byte,
        ncp string,
        tvu string,
        rpu string,
        tpu string,
        storage_addr string,
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
        leader_id: Pubkey.default(),
        ledger_state: LedgerState {
			last_id: Hash.default(),
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

	
func (nodeinfo * NodeInfo) new_with_pubkey_socketaddr(pubkey []byte, bind_addr string) NodeInfo{
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
	nodeinfo.new_with_pubkey_socketaddr(bind_addr)
}