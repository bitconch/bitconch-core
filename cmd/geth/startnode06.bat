RD /S /Q %~dp0\node00\geth
rem geth_BUS001  --datadir=node00 init genesis_clique.json
geth  --datadir=node00 init genesis_buffett.json
geth --identity "node00"  --nodiscover --rpc --rpcport "8545"  --port "30303" --networkid 1900 --datadir=node00  --rpccorsdomain "*" --mine --rpcapi "eth,web3,personal,net,miner,admin,debug" --ws  --wsaddr "0.0.0.0" --wsapi "eth,web3,personal,net,miner,admin,debug" --wsorigins "*" --unlock 0x12890d2cce102216644c59daE5baed380d84830c --password "pass.txt"  --verbosity 0 console  
