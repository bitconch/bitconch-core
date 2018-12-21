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

// Package crdt implements the Ethereum p2p network protocols.

package crdt


import (
    "os"
    "errors"
    godebug "runtime/debug"
    "fmt"
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
    "math/rand"
)

const (
	FULLNODE_PORT_RANGE = []uint16{8000, 10000}
	
	/// milliseconds we sleep for between gossip requests
	GOSSIP_SLEEP_MILLIS = 100
	GOSSIP_PURGE_MILLIS = 15000

	//minimum membership table size before we start purging dead nodes
    MIN_TABLE_SIZE = 2
    
    // default value of pubkey
    PUBKEY_DEFAULT_VALUE = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
)


func SocketAddr(ip string, port string) string {
	net.JoinHostPort(net.parseIPv4(ip), port)
}

func Socketaddr_any() string {
	SocketAddr(net.IPv4zero,'0')
}

type Crdt struct {
    /// table of everyone in the network
    table map[Pubkey]NodeInfo
        /// Value of my update index when entry in table was updated.
    /// Nodes will ask for updates since `update_index`, and this node
    /// should respond with all the identities that are greater then the
    /// request's `update_index` in this list
    local map[Pubkey]int64
    /// The value of the remote update index that I have last seen
    /// This Node will ask external nodes for updates since the value in this list
    remote map[Pubkey]int64
    /// last time the public key had sent us a message
    alive map[Pubkey]int64
    update_index int64
    id Pubkey
    /// last time we heard from anyone getting a message fro this public key
    /// these are rumers and shouldn't be trusted directly
    external_liveness map[Pubkey]map[Pubkey]int64
    /// TODO: Clearly not the correct implementation of this, but a temporary abstraction
    /// for testing
    scheduled_leaders map[int64]Pubkey
    // TODO: Is there a better way to do this? We didn't make this a constant because
    // we want to be able to set it in integration tests so that the tests don't time out.
    leader_rotation_interval int64,
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


func (nodeinfo *NodeInfo) New(
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

func (nodeinfo *NodeInfo) New_LocalHost(id Pubkey) NodeInfo {
	nodeinfo.new(
		id,
		SocketAddr("127.0.0.1","1234"),
		SocketAddr("127.0.0.1","1235"),
		SocketAddr("127.0.0.1","1236"),
		SocketAddr("127.0.0.1","1237"),
		SocketAddr("127.0.0.1","1238"),
	)
}
// NodeInfo with unspecified addresses for adversarial testing.
func (nodeinfo *NodeInfo) New_Unspecified() NodeInfo {
	addr = SocketAddr(net.parseIPv4,"0")
	nodeinfo.new(,addr, addr, addr, addr, addr)
}

func (nodeinfo *NodeInfo) Next_Port(addr , nxt int16) (SocketAddr string){
	host, port, _ = net.SplitHostPort(addr)
	socketaddr(host,ParseInt(port) + nxt)

	
func (nodeinfo *NodeInfo) New_With_Pubkey_Socketaddr(pubkey []byte, bind_addr string) NodeInfo{
	transactions_addr = bind_addr
	gossip_addr = nodeinfo.Next_Port(bind_addr,1)
	replicate_addr = nodeinfo.Next_Port(bind_addr,2)
	requests_addr = nodeinfo.Next_Port(bind_addr,3)
	nodeinfo.new(
		pubkey,
		gossip_addr,
		replicate_addr,
		requests_addr,
		transactions_addr,
		"0.0.0.0:0",
	)
}

func (nodeinfo *NodeInfo) New_With_Socketaddr(bind_addr string) NodeInfo{
	pub, _, _ := GenerateKey(rand.Reader)
	nodeinfo.New_With_Pubkey_Socketaddr(pub, bind_addr)
}

func (nodeinfo *NodeInfo) New_Entry_Point(gossip_addr string) NodeInfo{
	addr = socketaddr('0.0.0.0','0')
	pub, _, _ := GenerateKey(rand.Reader)
	nodeinfo.new(pub, *gossip_addr, daddr, daddr, daddr, daddr)
}



func (node *Node) New_LocalHost() Node {
	pub, _, _ := GenerateKey(rand.Reader)
	node.New_Localhost_With_Pubkey(pub)
}


func (node *Node) New_Localhost_With_Pubkey(pubkey []byte) Node {
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

func (node *Node) New_With_External_Ip(pubkey []byte, ncp &net.UDPAddr) {
	bind := func() (int16, net.UDPConn) {
		Bind_In_Range(FULLNODE_PORT_RANGE)
    }
    
    gossip_port, gossip = if ncp.Port != 0 {
        ncp.Port, Bind_To(ncp.Port)
    } else {
        bind()
    };

    replicate_port, replicate_sockets =
        Multi_Bind_In_Range(FULLNODE_PORT_RANGE, 8)

    requests_port, requests = bind()

    transaction_port, transaction_sockets =
        Multi_Bind_In_Range(FULLNODE_PORT_RANGE, 32)

    _, repair = bind()
    _, broadcast = bind()
    _, retransmit = bind()
    storage_port, _ = bind()

    // Responses are sent from the same Udp port as requests are received
    // from, in hopes that a NAT sitting in the middle will route the
    // response Udp packet correctly back to the requester.
    respond = requests.try_clone();

    info = NodeInfo.new(
        pubkey,
        net.UDPAddr{IP: ncp.IP, Port: gossip_port},
        net.UDPAddr{IP: ncp.IP, Port: replicate_port},
        net.UDPAddr{IP: ncp.IP, Port: requests_port},
        net.UDPAddr{IP: ncp.IP, Port: transaction_port},
        net.UDPAddr{IP: ncp.IP, Port: storage_port}
    )

    Node {
        info,
        sockets: Sockets {
            gossip,
            requests,
            replicate: replicate_sockets,
            transaction: transaction_sockets,
            respond,
            broadcast,
            repair,
            retransmit,
        }
    }
}

func Bind_In_Range(rang []int) (int16, net.UDPConn) {
    start, end = rang[0], rang[1];
    tries_left = end - start;
    for {
        rand.Seed(time.Now().Unix())
        rand_port = rand.Intn(end - start) + start
        addr = net.UDPAddr{IP: net.IPv4zero, Port: rand_port}
        con, error = net.UDPConn.DialUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: 0})
        if error == nil {
            return con.local_addr, con
        }
        if error != nil && tries_left == 0 {
            return error
        }
       
        tries_left -= 1;
    }
}


func Multi_Bind_In_Range(rang []int16, num uintptr) -> (int16, []net.UDPConn) {
    udpconns := make([]net.UDPConn, num)
    port, _ = Bind_In_Range(rang)

    for i := 0; i < num; i++ {
        conn, error = net.UDPConn.DialUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: port})
        udpconns = append(udpconns, con)
    }

    return port, udpconns
}

func Bind_To(port int16) net.UDPConn {
    conn, _ = net.UDPConn.DialUDP("udp", net.UDPAddr{IP: net.IPv4zero, Port: port})
    return conn
}


func (crdt *Crdt) New(node_info NodeInfo) Crdt {
    if node_info.version != 0 {
        return errors.New(fmt.Sprintf(CrdtError.BadNodeInfo))
    }
    me = Crdt {
        table: make(map[Pubkey]NodeInfo),
        local: make(map[Pubkey]int64),
        remote: make(map[Pubkey]int64),
        alive: make(map[Pubkey]int64),
        update_index: 1,
        id node_info.id,
        external_liveness: make(map[Pubkey]map[Pubkey]int64),
        scheduled_leaders: make(map[int64]Pubkey),
        leader_rotation_interval 100,
    }
    me.local[node_info.id] = me.update_index
    me.table[node_info] = node_info
    return me
}

func (crdt *Crdt) My_Data() &NodeInfo {
    return crdt.table[&crdt.id]
}

func (crdt *Crdt) Leader_Data() &NodeInfo {
    leader_id = crdt.table[&crdt.id].leader_id

    if leader_id == Pubkey.default(){
        return nil
    }

    return crdt.table[&leader_id]
}

func (crdt *Crdt) Node_Info_Trace() string {
    leader_id = crdt.table[&crdt.id].leader_id
    nodes = 
}

func (crdt *Crdt) Set_Leader(key Pubkey) {
    me = crdt.My_Data()
    me.leader_id = key
    me.version += 1
    crdt.insert(&me)
}

 // TODO: Dummy leader scheduler, need to implement actual leader scheduling.
 func (crdt *Crdt) Get_Scheduled_Leader( entry_height int64) Pubkey {
    match crdt.scheduled_leaders.get(&entry_height) {
        Some(x) => Some(*x),
        None => Some(crdt.My_Data().leader_id),
    }
}

func (crdt *Crdt) Set_Leader_Rotation_Interval(leader_rotation_interval int64) {
    crdt.leader_rotation_interval = leader_rotation_interval;
}

func (crdt *Crdt) Get_Leader_Rotation_Interval() -> int64 {
    return crdt.leader_rotation_interval
}

// TODO: Dummy leader schedule setter, need to implement actual leader scheduling.
func (crdt *Crdt) Set_Scheduled_Leader(entry_height int64, new_leader_id Pubkey) {
    crdt.scheduled_leaders[entry_height] = new_leader_id
}

func (crdt *Crdt) Get_Valid_Peers() -> []NodeInfo {
    var me := crdt.My_Data().id
    var nodeinfos = make([]NodeInfo)
    _, nodeinfo := range crdt.table {
        if nodeinfo.id != me && crdt.is_valid_address(&nodeinfo.contact_info.rpu) {
            nodeinfos = append(nodeinfos, val)
        }
    }
    return nodeinfos
}

func (crdt *Crdt) Get_External_Liveness_Entry(key &Pubkey) -> map[Pubkey]int64 {
    crdt.external_liveness.get(key)
}

func (crdt *Crdt) Insert_Vote(pubkey &Pubkey, v &Vote, last_id Hash) {
    if crdt.table[pubkey] == nil() {
        warn!("{}: VOTE for unknown id: {}", crdt.id, pubkey)
        return
    }
    if v.contact_info_version > self.table[pubkey].contact_info.version {
        warn!(
            "{}: VOTE for new address version from: {} ours: {} vote: {:?}",
            crdt.id, pubkey, crdt.table[pubkey].contact_info.version, v,
        )
        return
    }
    if *pubkey == crdt.My_Data().leader_id {
        info!("{}: LEADER_VOTED! {}", crdt.id, pubkey);
        inc_new_counter_info!("crdt-insert_vote-leader_voted", 1);
    }

    if v.version <= crdt.table[pubkey].version {
        debug!("{}: VOTE for old version: {}", crdt.id, pubkey);
        crdt.Update_Liveness(*pubkey);
        return;
    } else {
        var data := crdt.table[pubkey];
        data.version = v.version;
        data.ledger_state.last_id = last_id;

        debug!("{}: INSERTING VOTE! for {}", self.id, data.id);
        crdt.Update_Liveness(data.id);
        crdt.insert(&data);
    }
}


func (crdt *Crdt) Insert_Votes(votes &[(Pubkey, Vote, Hash)]) {
    inc_new_counter_info!("crdt-vote-count", votes.len());
    if !votes.is_empty() {
        info!("{}: INSERTING VOTES {}", crdt.id, votes.len());
    }
    for v in votes {
        crdt.Insert_Vote(&v.0, &v.1, v.2);
    }
}

func (crdt *Crdt) Insert(v: &NodeInfo) -> uintptr {
    // TODO check that last_verified types are always increasing
    // update the peer table
    if crdt.table[&v.id] ==nil || (v.version > self.table[&v.id].version) {
        //somehow we signed a message for our own identity with a higher version than
        // we have stored ourselves
        trace!("{}: insert v.id: {} version: {}", crdt.id, v.id, v.version);
        if crdt.table[&v.id] == nil {
            inc_new_counter_info!("crdt-insert-new_entry", 1, 1);
        }

        crdt.update_index += 1;
        _ = crdt.table.insert(v.id, v.clone());
        _ = crdt.local.insert(v.id, crdt.update_index);
        crdt.Update_Liveness(v.id);
        1
    } else {
        trace!(
            "{}: INSERT FAILED data: {} new.version: {} me.version: {}",
            crdt.id,
            v.id,
            v.version,
            crdt.table[&v.id].version
        );
        0
    }
}

func (crdt *Crdt) Update_Liveness(id Pubkey) {
    //update the liveness table
    now = timestamp();
    trace!("{} updating liveness {} to {}", crdt.id, id, now);
    *crdt.alive.entry(id).or_insert(now) = now;
}

/// purge old validators
func (crdt *Crdt) Purge(now uint64) {
    if len(crdt.table) <= MIN_TABLE_SIZE {
        trace!("purge: skipped: table too small: {}", crdt.table.len());
        return;
    }
    if self.leader_data() == nil {
        trace!("purge: skipped: no leader_data");
        return;
    }
    var leader_id := crdt.leader_data().id;
    var limit := GOSSIP_PURGE_MILLIS;
    var dead_ids = make([]Pubkey)
    k, v = range crdt.alive {
        if k != crdt.id && (now - v) > limit {
            dead_ids = append(dead_ids, k)
        }
    }

    inc_new_counter_info!("crdt-purge-count", len(dead_ids));
    for _, id := range dead_ids {
        delete(crdt.alive, id)
        delete(crdt.table, id)
        delete(crdt.remote, id)
        delete(crdt.external_liveness, id)
        info!("{}: PURGE {}", crdt.id, id);
        
        for _, v := crdt.external_liveness {
            delete(v, id)
        }
        if *id == leader_id {
            info!("{}: PURGE LEADER {}", crdt.id, id,);
            inc_new_counter_info!("crdt-purge-purged_leader", 1, 1);
            crdt.set_leader(PUBKEY_DEFAULT_VALUE);
        }
    }
}


/// compute broadcast table
/// # Remarks
func (crdt *Crdt) Compute_Broadcast_Table() -> []NodeInfo {

    //thread_rng().shuffle(&mut live);
    me := &crdt.table[&crdt.id];
    var cloned_table []NodeInfo
    for key, value = range crdt.alive {
        if crdt.table[key].id == me.id {
            //do nothing ,filter myself
        }
        else if crdt.is_valid_address(crdt.table[key].contact_info.tvu) {
            trace!(
                "{}:broadcast skip not listening {} {}",
                me.id,
                v.id,
                v.contact_info.tvu,
            )
        }
        else {
            cloned_table = append(cloned_table, )
        }
    }
   
    return cloned_table
}

/// broadcast messages from the leader to layer 1 nodes
/// # Remarks
/// We need to avoid having obj locked while doing any io, such as the `send_to`
func (crdt *Crdt) BroadCast(
    crdt: &Arc<RwLock<Crdt>>,
    leader_rotation_interval: uint64,
    me: &NodeInfo,
    broadcast_table: &[]NodeInfo,
    window: &SharedWindow,
    s: &UdpSocket,
    transmit_index: &mut WindowIndex,
    received_index: uint64,
) {
    if len(broadcast_table) == 0 {
        warn!("{}:not enough peers in crdt table", me.id);
        inc_new_counter_info!("crdt-broadcast-not_enough_peers_error", 1);
        Err(CrdtError::NoPeers)?;
    }
    trace!(
        "{} transmit_index: {:?} received_index: {} broadcast_len: {}",
        me.id,
        *transmit_index,
        received_index,
        len(broadcast_table)
    );

    old_transmit_index := transmit_index.data

    // enumerate all the blobs in the window, those are the indices
    // transmit them to nodes, starting from a different node. Add one
    // to the capacity in case we want to send an extra blob notifying the
    // next leader about the blob right before leader rotation
    //orders := Vec::with_capacity((received_index - transmit_index.data + 1) as usize);
    orders := make([]struct{Blob;NodeInfo}, received_index - transmit_index.data + 1)
    window_l := window.read().unwrap();

    br_idx = transmit_index.data % len(broadcast_table)
    
    for idx := transmit_index.data; idx <= received_index; idx++ {
        w_idx := idx % len(window_l);

        trace!(
            "{} broadcast order data w_idx {} br_idx {}",
            me.id,
            w_idx,
            br_idx
        );

        // Make sure the next leader in line knows about the last entry before rotation
        // so he can initiate repairs if necessary
        entry_height := idx + 1;
        if entry_height % leader_rotation_interval == 0 {
            next_leader_id := crdt.get_scheduled_leader(entry_height);
            if next_leader_id != nil && next_leader_id != me.id {
                var index uintptr
                for p, n = range broadcast_table{
                    if n.id == next_leader_id {
                        index = p
                        break
                    }
                }
                if index != nil {
                    // you can access the element inside orders by orders[index].Blob or orders[index].NodeInfo
                    orders = append(orders, struct{Blob;NodeInfo}{window_l[w_idx].data; &broadcast_table[index]})
                }
            }
        }
        orders = append(orders, struct{Blob;NodeInfo}{window_l[w_idx].data; &broadcast_table[br_idx]})
        br_idx += 1
        br_idx %= len(broadcast_table)
    }
    for idx := transmit_index.coding; idx <= received_index; idx++ {
        w_idx = idx % len(window_l);

        // skip over empty slots
        if len(window_l[w_idx].coding) <= 0 {
            continue;
        }

        trace!(
            "{} broadcast order coding w_idx: {} br_idx  :{}",
            me.id,
            w_idx,
            br_idx,
        );

        orders = append(orders, struct{Blob;NodeInfo}{window_l[w_idx].data; &broadcast_table[br_idx]})
        br_idx += 1;
        br_idx %= len(broadcast_table)
    }

    trace!("broadcast orders table {}", len(orders));
    errs :
    for b, v = range orders {
        if me.leader_id != v.id {
            blob := b.read()

            trace!(
                "{}: BROADCAST idx: {} sz: {} to {},{} coding: {}",
                me.id,
                blob.get_index(),
                blob.meta.size,
                v.id,
                v.contact_info.tvu,
                blob.is_coding()
            )
            if blob.meta.size <= BLOB_SIZE {
                return error
            }
            s, err := s.send_to(&blob.data[..blob.meta.size], &v.contact_info.tvu);
            trace!(
                "{}: done broadcast {} to {} {}",
                me.id,
                blob.meta.size,
                v.id,
                v.contact_info.tvu
            )
        }
    }

    trace!("broadcast results {}", errs.len());
    for e in errs {
        if let Err(e) = &e {
            trace!("broadcast result {:?}", e);
        }
        e?;
        if transmit_index.data < received_index {
            transmit_index.data += 1;
        }
    }
    inc_new_counter_info!(
        "crdt-broadcast-max_idx",
        (transmit_index.data - old_transmit_index) as usize
    );
    transmit_index.coding = transmit_index.data;

    Ok(())
}