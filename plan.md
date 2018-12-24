| Serial           | Package Name  | Lines of Code  |   Difficulty     | Lvl1 Package  | Lvl2 Package  | Issues |
| ------------- | :----------------: | :--------------: | :-------------: | :-------------:| :-------------:|   :-------------:| 
| 1 | crdt |  | Hard | |
| 2 | wallet |  | Hard | |
| 3 | bank |  | Hard | |
| 4 | erasure |  | Hard | |
| 5 | ledger |  | Hard | |
| 6 | fullnode |  | Hard | |
| 7 | thin_client |  | Hard | |
| 8 | window_service | 669 | Hard | |
| 9 | thin_client | 682 | Hard | |
| 10 | budget_program | 571 | Medium | |

| Serial           | Package Name  | Lines of Code  |   Difficulty     | Lvl1 Package  | Lvl2 Package  | Issues |
| ------------- | :----------------: | :--------------: | :-------------: | :-------------:| :-------------:|   :-------------:| 
| 11 | write_stage | 547 | Medium | |
| 12 | window | 566 | Medium | |
13 | packet
14 | banking_stage
15 | system_program
16 | broadcast_stage
17 | rpc
18 | drone
19 | choose_gossip_peer_strategy
20 | tvu

| Serial           | Package Name  | Lines of Code  |   Difficulty     | Lvl1 Package  | Lvl2 Package  | Issues |
| ------------- | :----------------: | :--------------: | :-------------: | :-------------:| :-------------:|   :-------------:| 
21 | metrics
22 | entry
23 | budget_transaction
24 | vote_stage
25 | wallet.1
26 | budget
27 | netutil
28 | sigverify
29 | dynamic_program
30 | streamer

| Serial           | Package Name  | Lines of Code  |   Difficulty     | Lvl1 Package  | Lvl2 Package  | Issues |
| ------------- | :----------------: | :--------------: | :-------------: | :-------------:| :-------------:|   :-------------:| 
31 | replicator
32 | recvmmsg
33 | counter
34 | system_transaction
35 | sigverify_stage
36 | result
37 | replicate_stage
38 | tpu
39 | request_stage
40 | signature



| Serial           | Package Name  | Lines of Code  |   Difficulty     | Lvl1 Package  | Lvl2 Package  | Issues |
| ------------- | :----------------: | :--------------: | :-------------: | :-------------:| :-------------:|   :-------------:| 
41 | retransmit_stage
42 | entry_writer
43 | poh_recorder
44 | rpu
45 | lib
46 | mint
47 | chacha
48 | ncp
49 | request_processor
50 | poh

| Serial        | Package Name       | Lines of Code    |   Difficulty     | Lvl1 Package   | Lvl2 Package   | Summary           | Referenced By # | Time
| ------------- | :----------------: | :--------------: | :--------------: | :-------------:| :-------------:| :----------------:| :-------------: |                
51              | store_ledger_stage | 73               |                  |                |                |                   | 45，31 
52              | hash               | 76               |                  |                |                |                   |
53              | fetch_stage        | 49               |                  |                |                |                   |
54              | storage_program    | 53               |                  |                |                |                   |
55              | budget_instruction | 38               |                  |                |                |                   | 
56              | blob_fetch_stage   | 48               |                  |                |                |                   |
57              | request            | 44               |                  |                |                |                   |
58              | payment_plan       | 28               |                  |                | ★              |                  
59              | timing             | 23               |                  | ★              |                | 4 functions 
60              | client             | 21               |                  |                | ★              | Create Client 
61              | logger             | 17               |                  | ★              |                | Create logging functionality    |
62              | service            | 8                | Easy             | ★              |                | Cretae a Service interface                 | 










**Lvl1 Package is a package only includes std and external package**