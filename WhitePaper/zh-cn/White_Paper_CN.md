#  BCOO CHAIN/贝克链
**A New Distributed Web Protocol Based on an Innovative Proof of Reputation (PoR) Consensus Algorithm and Eco System**

**基于创新性的信誉证明共识协议的分布式网络及生态系统**

作者: [BCOO.IO TEAM](http://www.bcoo.io)

**摘要**：贝克链提出了一种创新的PoR（Proof of Reputation）信誉共识算法，解决了目前区块链难以同时保持高吞吐量和去中心化两种特性的痛点。贝克链基于社交图谱，将社交、时间、贡献活跃度进行数学建模，构建了一套去中心化的信誉体系。每个用户都有机会建立高信誉值，用户信誉越高交易成本越低（甚至免费），被选为信任节点参与共识的机会越多，收益也越大。高信誉用户之间被定义为“互信节点”，小微交易将启动“支付通道”进行高速离线交易。

信誉体系和系统激励制度将有效促进商业开发者和普通用户的持续热情，有助于商业生态系统建设。拥有流量的商业开发者更容易获得高信誉值，当选可信全节点的几率更高。普通用户可以通过活跃的社交互动和积极使用生态系统中的商业应用来提升信誉值，增加被选为可信轻节点的几率。

贝克链采用了DAG有向无环图的数据结构以保持系统的正向可扩展性。支持智能手机轻节点客户端以抗击系统的去中心化，保持分散性。零知识验证、晶格化数据存储、量子级加密算法、改进的BVM虚拟机等技术，使贝克链具有更可靠的安全性，还提供了友好的DApp和侧链开发环境以满足某些应用在大文件存储、低交易成本、用户信息保护、侧链和智能合约迭代及Bug修复等方面的技术要求。

贝克链是一个无块无链的去中心化的分布式网络，解决了区块链在落地应用中的两个难点：可扩展性和分散性。贝克链，可以适用于千万级以上用户的商业应用需求，是目前最可行的高频小微交易和社交应用的区块链生态系统。

Copyright © 2018 bcoo.io

## 目录
<!-- MarkdownTOC depth=4 autolink=true bracket=round list_bullets="-*+" -->

- [1. 背景](#background)
- [2. 区块链在落地应用中的挑战](#chanllenges-for-blockchain-applications)
  * [2.1 高并发、高吞吐与可扩展性](#support-millions-of-users)
  * [2.2 商业信誉与用户隐私](#reputation-and-privacy)
  * [2.3 激励机制与交易成本](#incentive-mechanism-and-transaction-cost)
  * [2.4 安全性与去中心化](#security-and-decentralization)
  * [2.5 智能合约迭代](#upgrable-smart-contrct)
  * [2.6 存储能力受限](#storage-limitation)
- [3. 信誉共识](#proof-of-reputation-consensus-algorithm)
  * [3.1 信任与共识](#reputation-and-consensus)
  * [3.2 信誉是稀缺资源](#reputation-as-scarce-resources)
  * [3.3 信誉模型和数学抽象](#math-model-and-abstract)
    * [3.3.1 社交关系](#social-relationship)
    * [3.3.2 互信节点](#transaction-validators)
    * [3.3.3 信誉值量化](#quantization-of-reputation)
  * [3.4 共识过程](#consensus-process)
  * [3.5 诚信节点选择](#transaction-validator-list)
  * [3.6 拜占庭容错过程](#BFT-process)
  * [3.7 数据结构和交易关系](#datastructure-and-transaction-relationship)
  * [3.8 信誉证明的系统激励](#system-incentives-for-POR)
  * [3.9 PoR与各种共识机制的比较](#comparision-between-different-consensus-algorithms)
- [4. 其他技术](#other-breakthroughs)
  * [4.1 零知识验证/ZKP](#zero-knowledge-proof)
  * [4.2 轻节点/Thin Client](#thin-client-architecture)
  * [4.3 智能合约与分叉管理](#smart-contract-and-fork-mitigation)
  * [4.4 量子级加密算法](#quntum-proof-encrytpion)
  * [4.5 虚拟机和编程语言](#BVM-and-BO-Language)
  * [4.6 侧链支持](#side-chain-support)
- [5. 系统架构图](#system-architecture)
- [6. 落地应用场景展望](#realworld-application-outlook)
  * [6.1 轻节点——贝克钱包/BCOO PAY](#BCOOPAY-Wallet)
  * [6.2 其他可能的应用及技术支持](#Other-applications)
- [7. 结论](#conclusion)
- [8. 参考文献](#references)


<!-- /MarkdownTOC -->