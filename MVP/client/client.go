package client

import(
	"github.com/Bitconch/BUS/thin_client/ThinClient"
	"github.com/Bitconch/BUS/crdt"
	"time"
)

func (thinClient *ThinClient) MkClient(r crdt.NodeInfo) {
	_, requests_socket := bind_in_range(FULLNODE_PORT_RANGE);
	_, transactions_socket := bind_in_range(FULLNODE_PORT_RANGE);
	requests_socket.SetReadTimeout(time.Duration(1, 0));

	ThinClient.New(
		r.contact_info.rpu,
		requests_socket,
		r.contact_info.tpu,
		transactions_socket,
	)
}

