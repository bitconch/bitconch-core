![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/BCOO_logo.png "BCOO Logo")

#  BCOO CHAIN
**A New Distributed Web Protocol Based on an Innovative Proof of Reputation (PoR) Consensus Algorithm and Eco System**

**基于创新性的信誉证明共识协议的分布式网络及生态系统**

作者: [BCOO.IO TEAM](http://www.bcoo.io)

**Abstract**：Bitconch chain proposed an innovative POR (Proof Of Reputation) reputation consensus algorithm, which offers a new solution that leverage blockchain technology to maintain both high throughput and decentralization. According to social graphs, Bitconch blockchain mathematically models social network, time, and contribution activities to build a decentralized reputation system, which offer a chance to transform the above items into every single user’ reputation value. The higher the user's reputation, the lower the transaction cost (or even free of charge), and also has more opportunities to be selected as trust nodes to participate in the consensus and win better benefits. Users with high-reputation are defined as “Mutual Trust Nodes”, who can start “payment channels” for high-speed offline transactions through micro-transactions.

The reputation system and incentive system will effectively promote the continuous engagement of business developers and users, which also contribute to the construction of the business ecosystem. Business developers who generate traffic are more likely to get high reputation values, and better chances of being elected to a trusted full node. By actively engaging in social interactions or other commercial activities (via DApp on blockchain), users can increase their chances of being as Trusted Light Node, which will give user a privilege of sharing system reward.

Several technologies are used by the Bitconch blockchain to maintain the decentralization of the system, while increasing the scalability: DAG (directed acyclic graph) data structure, Zero-Knowledge-Proof, distributed data storage, post-quantum encryption algorithm, and BVM(Bitconch Virtual Machine, which is an enhanced virtual machine for smart contract). All these innovations make Bitconch chain a more reliable and developer friendly platform. DApp and sidechain developer can create awesome DApps which support large file storage, low transaction costs, user information protection, sidechain and smart contract iterations, and easy-bug-fixing. Bitconch chain is a decentralized network with no block and no chain, which solves two difficulties in the application of blockchain: scalability and decentralization. Bitconch chain, is the most feasible blockchain ecosystem for high-frequency micro-transactions and related applications on blockchain, which can be applied to the commercial application of more than 10 million users.

## Table of content
<!-- MarkdownTOC depth=3 autolink=true bracket=round list_bullets="-*+" -->

- [1. BACKGROUND](#1-BACKGROUND)
- [2. THE CHALLENGE OF BLOCKCHAIN IN APPLICATIONS](#2-THE-CHALLENGE-OF-BLOCKCHAIN-IN-APPLICATIONS)
- [3. PROOF OF REPUTATION CONSENSUS ALGORITHM](#3-PROOF-OF-REPUTATION-CONSENSUS-ALGORITHM)
- [4. OTHER BREAKTHROUGH](#4-OTHER-BREAKTHROUGH)
- [5. SYSTEM ARCHITECTURE DIAGRAM](#5-SYSTEM-ARCHITECTURE-DIAGRAM)
- [6. REAL-WORLD-APPLICATION SCENARIO OUTLOOK](#6-REAL-WORLD-APPLICATION-SCENARIO-OUTLOOK)
- [7. CONCLUSION](#7-CONCLUSION)
- [8. REFERENCES](#8-REFERENCES)


<!-- /MarkdownTOC -->

# 1. BACKGROUND

The birth of Bitcoin has made blockchain technology leap from pure theoretical research to the world's focal point of innovation and technology. Blockchain technology has initiated new ways to understand security and information usage and is now widely viewed as certain to dramatically change the world. The success of Ethereum and its Solidity language has permitted creation of Turing-complete smart contracts and allowed developers to create any kind of application to run on a blockchain.  However, the Merkle-tree architectures used by Ethereum, Bitcoin and others suffer from several limitations and these are what is holding them back from large-scale commercial adoption. These are:

- Simultaneous support for scalability, decentralization and security in one architecture. Only two are possible with Merkle-tree architectures
- Limited file storage
- Low Transaction Cost combined with Effective Incentives
- Upgradable Smart Contract
- Usability

# 2. THE CHALLENGE OF BLOCKCHAIN IN APPLICATIONS

## 2.1 HIGH CONCURRENCY, HIGH THROUGHPUT AND SCALABILITY

The core of business competition is volume competition. A successful business project will have more than 10 million registered users and more than one million active users.  Merkle-tree based blockchain architectures are severely limited in the speed and volume of transactions that can be processed. They are inherently not scalable without sacrificing their decentralization or security features. The fairly recent recognition of this has led to a burst of activity by these types of architectures (most notably Ethereum) to try to overcome these issues. Concurrently, the search for alternative architectures that support the decentralization, non-repudiation, and security features of Merkle-tree architectures is a source of huge technical activity and innovation. In the commercial world, the ability to support system throughput in the range of tens of thousands of transactions per second –the current volume and speed Visa/MasterCard support – is expected of any potential replacement technology.

## 2.2 REPUTATION AND USER PRIVACY

Blockchain technology was hailed for its anonymity. By hiding the identity of the users, Blockchain can protect the peers’ privacy. However, in real business applications, pure anonymity may bring problems such as fraud, breach of contract, and difficulty in defending rights. When a user chooses a service provider, they have the right to know whether the service provider is honest and trustworthy.

## 2.3 INCENTIVE MECHANISM AND TRANSACTION COSTS

The business application scenario is mainly for high-frequency small and micro-transactions for a range of small and medium-sized users, so transaction costs will become an important consideration. The transaction cost of Bitcoin has exceeded $1/transaction and the transaction cost of Ethereum is 0.01~0.02ETH/transaction, which is about 5~10 USD/transaction. Excessive transaction costs are clearly unable to meet the commercial needs of high-frequency micro-transactions.

Consumers may wish to use the resources on the blockchain platform for free or at low price, but there is a clear conundrum for a decentralized system: there is no central authority to maintain the cost of transacting on the system, so it is essential to have an effective incentive mechanism which keeps the peers – nodes performing transactions – participating in the activities that keep the system as a whole working. The peers therefore need adequate incentive to provide payment for operation costs, including equipment and utility fees. The participation and therefore the type and degree of incentives for peers is key to operating a truly self-sustaining decentralized system. How to effectively balance the incentive needs for operating the system and secure and genuine decentralization is key to a sustainable platform.

## 2.4 SECURITY AND DECENTRALIZATION

In order to ensure transaction security, the clients need to download and backup all the transaction data of the whole network, these clients are called “Full Node”. However, running a full node in most cases is extremely expensive and slow, and most users in commercial applications are dealing with small micro-transactions, and have no ability or demand to purchase large computers and bear the corresponding operating costs. Therefore, small medium-sized users are effectively blocked from participating in system computing process (consensus process etc.) and cannot obtain system rewards, thus forming a monopoly of computing power for a small number of rich users and potentially compromising the consensus mechanisms supporting security.

## 2.5 UPGRADABLE SMART CONTRACT

Applications developed on the blockchain need to have an effective mechanism to support App upgrade. All App can be affected by bugs. When a Side Chain or DApp encounters a bug, it needs to be able to fix the errors from the bug.

## 2.6 LIMITATION ON STORAGE

Most development of many commercial applications involves large file storage and transmission. For example, applications such as media, social software, e-commerce platforms, live video, games, etc. require large file storage functions. The existing blockchain does not support the storage of large files, and cannot meet the demands of real-world applications.

# 3. PROOF OF REPUTATION CONSENSUS ALGORITHM

The POR reputation consensus algorithm is a new algorithm proposed by Bitconch based on social graphs. Using a distributed decentralized reputation quantification system built on the blockchain, POR can show the contribution of each peer to the whole network, including growth, security, and stability. By introducing the “R” vector which would mathematically represent the reputation of each peer, POR can maintain a list of trustworthy peers, which is called Transaction Validators. These Transaction Validators will be rewarded for their contribution to the system by providing computing power or storage capabilities.

The POR utilize a directed acyclic graph as the basis of the social graph. The POR can tolerate Byzantine failures. Bus is the underlying token powering Bitconch blockchain, as part of Bitconch protocol, POR can help Bus to scale: the greater the number of peers, the greater the number of transaction per second (TPS). POR is a perfect candidate for high-frequency small micro-transactions and social oriented DApps. POR adopts the power balance philosophy and designs a decentralized incentive system with anti-Matthew effect, which can ensure that new peers and existing peers, small users and large users have the same opportunity to obtain system rewards. POR can prevent the system from becoming centralized that are a result of poorly designed incentive mechanisms.

## 3.1 REPUTATION AND CONSENSUS

As advocated by the blockchain pioneers such as CypherPunk, HashCash, and B-Money, the ideal blockchain is a decentralized system, a system with no central authority. An ideally decentralized system eliminates the potential threat of corruption and abuse caused by the concentration of control. The core of much of the blockchain technology is the consensus algorithm. The essence of the consensus algorithm is that in a distributed network and under the condition that each node does not trust each other, the Nash equilibrium is formed by the evidence of scarce resources, winning the trust of all parties, thus an agreement is achieved among the nodes and permit the task to will be completed synchronously.

For POW (Proof of Work), the time and electricity consumed by the miners will be considered as evidence. For POS (Proof of Stake) or DPOS, the value-vector (coins or coin-age) will be considered as evidence. The consensus process is about finding someone who can be proved to hold a scarce resource as evidence and thus to be trusted to participate in the consensus process and thereby contribute computing power, and receive rewards.

Bitconch proposes that consensus can be reached by means of proof of reputation.

## 3.2 REPUTATION AS SCARCE RESOURCE

The business community relies on reputation and reputation markers to help evaluate the trustworthiness and creditworthiness of many potential counterparties, such as mortgage applicants and other loan transactions, trading financing, etc. On a blockchain, reputation can help peers to decide who is trustworthy to participate in the consensus and who should get the reward for maintaining the system. Reputation in existing business environment can be earned by paying back on time. On blockchain, reputation can be earned by maintaining the distributed ledger honestly and restrained from perpetrating malicious activities.

Reputation can be considered as a scarce resource, and in real life, reputation can be quantified. On many e-commerce platforms such as Taobao or Amazon, users rely on centralized ratings from platforms (stars for Amazon or Diamonds/Crowns for Taobao) to decide which product to buy. In the investment market, users rely on centralized ratings from prestigious institutions such as Standard & Poor's and Moody's. When choosing a university, students use the reputation and ranking of the school to decide which one he or she should apply to. The accumulation of reputation requires time and effort, and the level of reputation and economic value also have a natural connection. The famous investor Warren Buffett has a famous saying, “It takes 20 years to build a reputation and 5 minutes to ruin it. If you think about that, you'll do things differently.” Because of the scarcity and high value of reputation, using a reputation value as evidence of validity can effectively reduce and manage the possibility of malicious nodes.

The Bitconch chain uses a social graph and mathematical abstraction built by commercial activities to construct a quantifiable reputation system. It is fair for or anyone or an organization, hard to forge, and quantifiable. Anyone or organization can build and maintain its own reputation.

## 3.3 MATHEMATICAL MODEL FOR REPUTATION

Social network can be mathematically abstracted and modeled, and a Social Graph can be constructed. Participants in the network can be abstracted into points (Vertices/Node), and the relationship between participants can be abstracted into edges of the graph. We can use mathematical descriptions to describe the social relationships, intimacy, and personal credibility between people.

### 3.3.1 SOCIAL RELATIONSHIP

Assuming that a and b are two people, they can be represented as two vertices Va and Vb in the figure. If a transaction occurred between a and b, a line can be drawn between vertices a and b, as E(a,b) which means the direction is from a to b. Also, We define the weight of E(a,b), along with the amount and number of transactions that can affect the weight value.




