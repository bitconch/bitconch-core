![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/BCOO_logo.png "BCOO Logo")

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

- [1. 背景](#1-背景)
- [2. 区块链在落地应用中的挑战](#2-区块链在落地应用中的挑战)
  * [2.1 高并发、高吞吐与可扩展性](#21-高并发、高吞吐与可扩展性)
  * [2.2 商业信誉与用户隐私](#22-商业信誉与用户隐私)
  * [2.3 激励机制与交易成本](#23-激励机制与交易成本)
  * [2.4 安全性与去中心化](#24-安全性与去中心化)
  * [2.5 智能合约迭代](#25-智能合约迭代)
  * [2.6 存储能力受限](#26-存储能力受限)
- [3. 信誉共识](#3-信誉共识)
  * [3.1 信任与共识](#31-信任与共识)
  * [3.2 信誉是稀缺资源](#32-信誉是稀缺资源)
  * [3.3 信誉模型和数学抽象](#33-信誉模型和数学抽象)
    * [3.3.1 社交关系](#331-社交关系)
    * [3.3.2 互信节点](#332-互信节点)
    * [3.3.3 信誉值量化](#333-信誉值量化)
  * [3.4 共识过程](#34-共识过程)
  * [3.5 诚信节点选择](#35-诚信节点选择)
  * [3.6 拜占庭容错过程](#36-拜占庭容错过程)
  * [3.7 数据结构和交易关系](#37-数据结构和交易关系)
  * [3.8 信誉证明的系统激励](#38-信誉证明的系统激励)
  * [3.9 PoR与各种共识机制的比较](#39-por与各种共识机制的比较)
- [4. 其他技术](#4-其他技术)
  * [4.1 零知识验证/ZKP](#41-零知识验证zkp)
  * [4.2 轻节点/Thin Client](#42-轻节点thin-client)
  * [4.3 智能合约与分叉管理](#43-智能合约与分叉管理)
  * [4.4 量子级加密算法](#44-量子级加密算法)
  * [4.5 虚拟机和编程语言](#45-虚拟机和编程语言)
  * [4.6 侧链支持](#46-侧链支持)
- [5. 系统架构图](#5-系统架构图)
- [6. 落地应用场景展望](#6-落地应用场景展望)
  * [6.1 轻节点——贝克钱包/BCOO PAY](#61-轻节点——贝克钱包bcoo-pay)
  * [6.2 其他可能的应用及技术支持](#62-其他可能的应用及技术支持)
- [7. 结论](#7-结论)
- [8. 参考文献](#8-参考文献)


<!-- /MarkdownTOC -->

# 1. 背景
比特币的诞生使区块链技术从单纯的理论研究一跃成为全世界瞩目的创新科技，被寄予厚望，希望通过“区块链改变世界”。以太坊提出智能合约，使区块链的落地应用成为可能。但是，由于众多技术瓶颈的限制，目前区块链距离大规模的商业落地应用可能还需要解决以下技术瓶颈问题：
- 可扩展性
- 保持去中心化
- 文件存储
- 低交易手续费和有效激励
- 智能合约可迭代
- 用户信息保护
- 系统安全性
- 易用性

# 2. 区块链在落地应用中的挑战
## 2.1 高并发、高吞吐与可扩展性
商业竞争的核心是流量竞争。一个成功的商业项目通常要拥有千万以上的注册用户量和百万以上的活跃用户。区块链的可扩展性问题，即如何提高交易的速度和交易的吞吐量是目前区块链研究的重点。生态系统中多个应用（DApp）可能产生高并发请求，系统吞吐量应至少满足每秒处理数万笔交易，达到或接近Visa/万事达卡的水平。
## 2.2 商业信誉与用户隐私
区块链在某些领域因为匿名性而受到追捧。匿名性在于隐藏了用户的身份，将用户信息进行隐私性保护，但在真实的商业应用中，单纯的匿名性可能带来欺诈、违约、难以维权等问题。当用户选择服务商时，有权利知道该服务商是否诚实可信。
## 2.3 激励机制与交易成本
商业应用场景主要是面对中小用户的高频小微交易，因此交易成本将会成为项目落地应用的重要考虑因素。比特币的交易成本已经超过1美元/次，以太坊的交易成本在0.01~0.02ETH/次，约合5~10美元/次。过高的交易成本显然无法满足高频小微交易的商业需求。

C端用户希望免费或低成本使用平台资源，但记账节点需要足够的激励来支付设备运行成本，并保持持续的参与热情，以维持去中心化系统的分散性。如何有效平衡两者的矛盾，是建立一个可持续发展的底层系统的关键。

## 2.4 安全性与去中心化
经典区块链为了确保交易安全性，需要每个客户端都下载备份全网的交易数据，称为“全节点”。但多数情况下运行全节点是极其昂贵和迟缓的，对于商业应用中的大多数用户都在处理小微交易，并没有能力与需求购置大型计算机和承担相应的运行费用。因此，中小用户无法参与系统运算，无法获得系统奖励，从而形成了少数用户的**算力垄断**。

## 2.5 智能合约迭代
区块链上开发的应用程序在进行功能迭代时需要有一套合理的机制支持软件升级。所有软件都有可能受到bug的影响，当一个区块链底层平台或运行的应用遭遇bug的时候，需要能够从bug中修复错误。

## 2.6 存储能力受限
很多商业应用产品的开发往往涉及大文件存储和传输，例如自媒体、社交软件、电商平台、视频直播、游戏等应用都需要较大的文件存储功能。经典区块链不支持存储大文件，无法满足区块链在实际落地应用中的开发需求。

# 3. PoR信誉共识
PoR信誉共识算法是贝克链基于社交图谱提出的崭新算法。利用在区块链上构筑的分布式去中心化的信誉量化体系，表明每个用户对全网发展性、安全性和稳定性的贡献。根据R值动态筛选具有可信赖度的诚信节点，诚信节点通过贡献算力，维持系统的完整性，可以获得系统奖励。

信誉共识在社交图谱的基础上结合了DAG有向无环图的数据结构，具有拜占庭容错功能。贝克链具有正向可扩展性，用户越多交易速度越快，可适用于高频小微交易和社交应用。信誉证明采用了权力制衡哲学，设计了一个具有防马太效应的去中心化激励制度，可以确保新用户和老用户，小用户和大用户之间，都有机会获得系统的奖励，防止因为系统激励而导致的中心化。

## 3.1 信任与共识
理想的区块链是不存在中央权威的去中心化系统，如在CypherPunk，HashCash，B-Money等区块链先驱所倡导的那样杜绝权力集中而导致的腐败和滥用。区块链技术的核心是共识算法，共识算法的本质是在分布式网络中，各节点互不信任的条件下，通过举证稀缺资源的方式，形成了纳什均衡，赢得各方的信任，快速在各个节点之间达成一致，并同步的完成任务。

PoW，Proof of Work/工作量证明，举证的是矿工的时间和电力；PoS，Proof of Stake/权益证明和DPoS，Delegated Proof of Stake/代理人权益证明，举证都是权益，用持币数量或币龄计算。共识就是能够举证某项稀缺资源，因而被信任，可以参与共识，贡献算力，获得奖励。

贝克链提出通过举证信誉的方式，来获得共识的达成。

## 3.2 信誉是稀缺资源
传统商业社会的信誉是解决银行把钱借给谁、房东把房子租给谁的问题，区块链中的信誉解决选择谁来参与共识，谁该获得记账奖励的问题。传统商业社会的信誉通过按时还款来证明自己的信誉，区块链通过准确记账、不作恶来证明自己的信誉。

贝克链认为信誉是一种稀缺资源。在电商平台上，用户依靠商家的钻数或星级来决定购买哪家的商品。在投资市场，用户通过标准普尔、穆迪等机构的评级来决定购买哪家的债券。在选择大学时，学生利用学校的声望和排名来决定自己报考志愿。信誉的积累需要时间和努力，而且信誉的高低和经济价值也有天然的联系。如巴菲特的名言“二十年积累的信誉会在五分钟毁于一旦，知晓此事，我们每个人都会小心行事”。因为信誉的稀缺性和高价值，将信誉值作为举证物，可以有效降低节点作恶的可能性。

贝克链利用商业应该构建的社交图谱，用数学抽象构造出可量化的信誉体系。任何人或机构建立和维持自己的信誉是公平的、不易的、可量化的。

## 3.3 信誉模型和数学抽象
对社交网络，Social Graph，进行建模和数学抽象。每个人可以被抽象成一个个的点（Vertices/Node），人与人之间的关系可以抽象成为图的一个边（Edge）。我们可以用数学描述图的方式来描述人与人之间的社交关系、亲密程度以及个人信誉度。

### 3.3.1 社交关系
假设a和b为两个人，他们可以表示为图中的两个顶点Va和Vb。如果a和b发生了交易行为，则可以定为（a,b）即从a到b的一条边，E(a,b)，我们将交易的数量和次数表示为边E(a,b)的权重。

### 3.3.2 互信节点
对于小微交易，互信节点可以开启“交易通道”执行高频离线交易。互信节点包括两种类型，亲友型互信节点和高信誉互信节点。

如果两个节点之间的社交互动频繁，在区块链底层将反映为交易次数频繁，则可认为这两个点具有深度的社交关系。如图所示，Jenny和Alice是朋友，她们经常通过社交工具进行互动，例如：加密微信聊天、传图片、发红包/打赏、自媒体互评等。贝克链定义Jenny和Alice是具有深度社交关系的两个节点，称为“亲友型互信节点”，适用于高频离线交易。

本身具有高信誉值的两个节点即便是第一次进行小微交易也将开启“交易通道”执行高频离线交易。例如具有高信誉的Tom和Jack被视为“高信誉互信节点”，他们虽然是第一次交易，但由于各自在自己的社交网络中拥有良好的信誉表现，对于小微交易来说，作恶之后信誉损失的成本远远大于可能的收益。

### 3.3.3 信誉值量化
我们定义信誉值R是个体在一个社交网络中被认可的程度。在区块链网络中，我们将信誉R由三个维度构建：社交活跃度D、时间活跃度T和贡献活跃度C。公式如下: 

其中 ω_n为权重，在某个时间t内，D(α,t)为节点的社交活跃度，T(β,t)为每个节点的时间活跃度，C(γ,t)为贡献活跃度。为了能让用户有持续不断的活跃度，同时也为了让后来者能更加公平的参与系统运行，避免先行者优势（FMA）带来的马太效应，我们规定R随着时间而进行衰减。声誉值R 随着时间衰减率为μ，如公式（2）所示：

社交活跃度D：由一个节点在应用社交网络中朋友的数量、与朋友互动的频率（即热度）、朋友的信誉值和交易额大小等多种因素决定。公式如下: 

其中E_i为每次交易的权重函数，E_i与交易金额正相关，D_r是交易对象，log⁡(D_r)为关于D_r信誉值的对数函数，用于控制当D_r过小时，即一个节点和一个或者多个低信誉度交易，并不能显著提高自己的D值，这有效规避了试图通过增加虚假用户恶意提升信誉的可能。

如图所示：Tom只有少数几个朋友，而且都是低频次单向交流。而Jack有很多朋友，与朋友之间也是频繁互动的，有的朋友是高信誉用户，与有的朋友之间互为可信节点，适用于“交易通道”的高频离线交易。则Jack的D值远远高于Tom的D值。

时间活跃度T：该指标主要由用户持有BCOO的币龄决定，我们认为通证的长期持有者比非持有者更可信，作恶动机更小。但与PoS共识中Stake权益不同，财富并非衡量节点是否可信的唯一标准。如图所示，T(β,t)的对数公式为广大的中产阶级用户提供获得高信誉机会。公式如下：

贡献活跃度C：该指标描述节点用户对于系统的贡献度C(γ,t)，表示在时间为t时，节点对于系统做了多少的贡献，N是系统Account Nonce值，用于记录用户对于系统贡献的频率（分享文件和参与记账）。系统将会按定时任务对文件状态进行检查。

## 3.4 共识过程
信誉证明的共识过程分为两部分：

（一）定义可信节点列表

（二）通过拜占庭容错过程验证交易

定义诚信节点（Transaction Validator），在区块链网络中，有N个用户，每个用户都有信誉值R，根据每个节点的信誉R筛选出包含有n个高R值的节点，标记为列表L。

诚信节点分为全节点候选人和轻节点候选人。全节点的用户可能来自于贝克链生态中的商业开发者或其他社群机构。由于轻节点可以运行在智能手机和家用电脑等设备上，所以几乎所有的用户都可以成为轻节点候选人。全节点和轻节点共同记账的方式不仅可以激发商业开发者和普通用户双方的积极性，还能制约可能出现的中心化倾向。

L中的诚信节点通过拜占庭容错过程，验证网络中的新交易。验证成功的交易被记录到系统的分布式账本之中，同时增加相应节点的信誉值。拜占庭容错过程，可以在有限作恶节点存在时，仍然为系统提供安全性和活性的保证。

## 3.5 诚信节点选择

诚信节点列表中L的节点，轮流参与验证共识。每一轮都会运行拜占庭容错协议来验证新生成的交易。如果L中的所有节点，均达成一致，则交易获得确认。

L根据每个节点当前的R值和一个分布函数D来选择。对于节点x来说，他的R_x越高，则越容易被记录到L中。

用P表示一个节点被选入L的概率，则有如下：

选择方法：（1）根据所有节点的R降序排序（2）根据D，在[0,├ N)┤生成m个随机数，然后向下取整。为了防止取重复，根据最近的和高信誉值的未选择节点，依照顺序从高信誉节点向低信誉节点选择。 

分布函数D是一个指数型分布函数，相比较于三角分布，构造指数分布可以给高信誉的节点更高的机会，并且压制低信誉的节点。

## 3.6 拜占庭容错过程
在获得L之后，信誉证明通过一个拜占庭容错过程验证交易的正确性并更新账本。我们做出如下定义：

**非故障节点：**

系统中运行正常，遵守规则和无错误的节点。

**故障节点：**

出现错误的节点，包括超时、数据损坏和恶意行为（拜占庭错误）等。
我们做出如下规定：

- 节点使用1或者0表示交易验证结果。0为验证成功，1为验证失败。
- 所有的诚信节点会在有限时间内做出决策。
- 所有的诚信节点会做出相同决策。

信誉证明将共识分为若干个周期，每个周期有若干回合，每回合会处理若干交易数据，因此可定义L(k)为第k回合的诚信节点列表。

达成共识流程：

- 上一回合结束。
- 有若干未验证交易Tx0。
- 本回合又有若干新交易生成Tx1。 
- 有Tx0和Tx1组合成待验证交易列表Tx。
- Tx会在L(k)中广播，L(k)中的节点会验证交易，若交易获得足够多的节点验证，则该交易会被更新到账本中。
- L(k)的作恶节点小于m/3，则该回合定义为成功。L(k)中节点的贡献活跃度增加，相应节点的交易活跃度增加。

## 3.7 数据结构和交易关系

贝克链采用一种无块无链的DAG有向无环图的数据结构。如图1（a）所示，Tx0为创世交易，即整个网络在运行时的第一笔交易，由一条特殊地址向初始用户分配BCOO。Tx1，Tx2，Tx3，Tx4，Tx5，Tx6，…，Txn为后续的交易。因为每笔交易存在时间（Time）和顺序（Order），构成DAG有方向且无环图。

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-7-1-1a.png "图1（a）DAG数型结构")

图1（a）DAG数型结构

Tx0是一条为节点N1充值的交易记录，对应着图1（b）中有了第一个用户N1。Tx1为第二笔交易N1N2，即N1向N2转账若干BCOO，社交图谱中用户N1至N2将增加一条边，即N1和N2开始建立起社交关系。随着交易Tx的增多，社交图谱中各个节点之间会有越来越多的边，社交网络趋向成熟。

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-7-1-1b.png "图1（b）社交图谱")

图1（b）社交图谱

图1（a）和（b）展示了DAG数据结构和社交图谱之间的互动关系。15个用户产生了从Tx0，Tx1到Tx13的14笔交易，构建了如（b）所示的社交关系。

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-7-1-2a.png "图2（a）新交易产生及验证")

图2（a）新交易产生及验证



如图2（a）所示，新交易Tx14和Tx15生成。其中Tx14表示N1向N4转账m个BCOO，Tx15表示N5向N1转账n个BCOO，如果m>n，根据公式（3），由于交易额度与E值正相关，对于N1的信用贡献，Tx14大于Tx15的权重。随着交易的不断增多，社交图谱中各个节点之间的联系不断增加，提供更多的社交数据喂养信誉值。


![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-7-1-2b.png "图2（b）新交易对社交图谱的影响")

图2（b）新交易对社交图谱的影响

图2（a）还演示了系统处理并发交易的能力，当Tx14和Tx15同时产生时，系统可以并发多个拜占庭容错过程，提高交易验证的效率。

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-7-1-3.png "图3 双花交易")

图3 双花交易

如图3所示，若系统出现双花交易Tx16和Tx17（N1作恶），由于拜占庭容错过程的确定性特质，即使Tx16和Tx17被同时确认，当其中的一个被更新到账本上后，另一个由于余额不足就会被自动舍弃，从而避免了双花攻击的出现。

作恶的N1节点将被追溯并收到惩罚，信誉值降低，失去参选信任节点的资格。因为N1的作恶成本远高于作恶可能带来的收益，所以作恶动机极低。



## 3.8 信誉证明的系统激励

贝克链主网将产生500亿枚BCOO通证，不需要挖矿。其中20%即100亿枚BCOO将作为PoR信誉共识的初始奖励池。所有全节点和轻节点都有机会被选中参与共识，获得奖励。

奖励池除了100亿枚初始BCOO以外，还将通过商业生态系统中获得的收益进行源源不断的补充，例如：交易手续费、应用开发费、推广收入等。可持续的奖励池能够保持节点的参与热情，有助于生态系统的可持续健康发展。

贝克链包含全节点和轻节点，由用户根据自己的设备和兴趣自主选择。全节点和轻节点将按照3:2的比例分享系统奖励的BCOO。

持有早期ERC20 BCOO通证（发行总量10亿枚）的用户可以在主网上线后按1:50的比例映射迁移，兑换成主网通证。

**PoR共识规则小结：**
第一、贝克链采用类以太坊的账户架构，每个用户都有一个账户，账户中记载了每个用户的BCOO余额、信誉值等数据。

第二、互信节点用户之间可开启小微交易支付通道，进行离线高速交易。

第三、通过R值动态筛选具有可信赖度的诚信节点，形成列表L。

第四、诚信节点通过贡献算力，维持系统的完整性，可以获得系统奖励。

第五、互为可信任节点之间的大额交易，以及非互相可信任节点之间的交易，由诚信节点验证。

## 3.9 PoR与各种共识机制的比较
由下表对比可知，PoR信誉共识机制不仅在支持高吞吐量和高并发量方面表现卓越，且具有正向可扩展性，用户越多交易速度越快。在保持低使用手续费的同时，通过信誉激励维持全民参与的积极性，保持全网的分散性。

区块链平台 | BTC | ETH | EOS | Ada(Cardano) | IOTA | BCOO
------------ | ------------- | ------------- |  ------------- |  ------------- |  ------------- |  ------------- 
共识算法 | PoW/工作量证明 | PoW/工作量证明 | DPoS/代理人权益证明 | Ouroboros算法/权益证明 | DAG+Tangle | PoR/信誉证明
举证物 | 时间+电力 | 时间+电力 | 币量+投票 | 币量 | 时间 | 信誉
违约惩罚 | 工作量被忽略，无回报 | 工作量被忽略，无回报 | 失去代理人资格 | 失去记账资格 | 工作量被忽略 | 信誉降低，失去记账资格
容错机制 | 1/2 + 1 | 1/2 + 1 | 1/3 + 1 | 1/2 + 1 | 1/3+1 | 1/5 + 1
吞吐量 | 3~7 | 5~20 | 1,000 | 257 | 800 | 10,000+ 
高并发量 | 不支持 | 不支持 | 支持 | 支持 | 支持，正向可扩展性 | 支持，正向可扩展性
分散性 | 算力集中在矿场 | 算力集中在矿场 | 中心化，集中在超级节点 | 集中于高币量玩家 | 去中心化 | 去中心化
交易成本 | 高 | 高 | 低 | 低 | 低 | 低 
矿工积极性 | 高 | 高 | 高，但仅限超级节点 | 低 | 低 | 高，全民参与

 ![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-3-9-1.png "BCOO可扩展性")

# 4. 其他技术
## 4.1 零知识验证/ZKP
贝克链通过零知识验证将交易详情进行加密，从而保护用户的隐私。传统区块链，如比特币和以太坊，分布式账簿在互不信任的网络节点之间大规模复制，链上存储的信息是完全公开的。即使用户每次都使用新的地址，攻击者仍然可以通过分析用户习惯、消费金额、交易时间等信息来判定用户的真实身份。

零知识验证通过对于交易细节进行加密，从而达到保护用户隐私的作用。零知识验证采用一种零知识证明架构，证明者可以在不表露信息内容的情况下，向验证者证明自己拥有某种信息（如加密密钥中的私钥）。

## 4.2 轻节点/Thin Client
传统区块链的客户端随着网络节点的不断增长，客户端系统需要耗费更大的存储空间，运行速度迟缓，成本昂贵，已超出了个人电脑可以承受的范围。不可避免的导致大多数普通用户难以参与到系统记账中，资源和算力更加集中到少数参与者手上，形成马太效应。

贝克链通过系统快照（Snapshot）和分布式哈希表两种方式实现节点的轻量化。简化的轻节点，可以在配置更低的设备上运行，包括智能手机、家用电脑等，参与系统记账。数以千万计的轻节点有利于抗击中心化，保障系统的分散性。

## 4.3 智能合约与分叉管理
贝克链提供套件可以让用户生成可修改的智能合约模板。用户根据贝克链提供的模板和规则，可以开发出易于升级和管理的智能合约。

由于区块链数据不可更改的中心思想，任何对于区块链上的修改都有潜在的分叉可能，因此对于具有重大利益相关的智能合约或者类似的更改，贝克链采用投票的方式：利益相关方在一定时间用自己的Stake（通证）进行投票。

## 4.4 量子级加密算法
比特币和以太坊等传统区块链都使用了非对称性加密算法（Asymmetrical），如ECC256、SHA256、SHA3等，这些加密算法容易被量子计算机破解。

目前被认为可以抵抗量子攻击的密码学系统包括：哈希密码系统、编码密码系统、晶格密码系统、多变量二次密码系统以及密钥密码系统。上述密码系统，在密钥长度足够长的前提之下，都可以同时抵抗经典和量子攻击。为了应对将来量子计算机时代的到来，贝克链将采用抗量子攻击的密码学算法。

## 4.5 虚拟机和编程语言
贝克链通过提供多种工具支持开发者打造属于自己的分布式应用，丰富贝克链的生态系统。贝克链提供了基于Solidity的编程语言BO和相对应的虚拟机BVM，如图所示。在贝克链上，开发者通过编程语言将商业逻辑转化为智能合约，智能合约通过虚拟机（Virtual Machine），将编程语言编程成机器可以运行的字节码。

BVM相对于EVM有三大优势：

第一、更容易开发强大功能的智能合约

相比较以太坊的65个opcodes，贝克链为了方便开发者能开发更优质的DApp，会提供更多可选的opcodes和标准库，扩展更多的社交和落地应用的功能。因为智能合约中通常都有大量的代币存在，一旦出现错误将对开发者和用户造成巨大损失，因此BVM将会提供智能工具，检测交易顺序、时间戳、意外处理和可重入漏洞（Reentrancy Vulnerability）等常见Bug。为了提升开发速度，让开发者更容易编写智能合约，BVM将是一个基于寄存器的虚拟机。

第二、提供接口，使智能合约和外部进行通信

相对于EVM和外部世界隔离（无法使用网络、文件或者其他进程的权限）的沙箱环境，BVM通过数字签名建立传输通道，解决智能合约和外部世界的通信问题。

第三、支持多语言开发

为了能让更多的开发者加入贝克链社区，BVM未来将支持Python，Java，C++等多种开发语言。

 ![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-4-5-1.png "BVM虚拟机")

## 4.6 侧链支持

贝克链支持不同的挂钩机制（Two-way Peg/双向锚地）实现主链和侧链的结合，并将为开发者提供侧链开发模板。在未来的应用场景中，贝克链作为主链将主要提供可信记账和信誉管理，更丰富的商业功能将开放并支持侧链进行实施。例如：贝克链将提供分布式存储功能，开发者可以在自己的侧链上实现文件存储、多媒体等功能。

# 5. 系统架构图

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-5-1.png "系统架构图")

注：▲项为采用了当前最先进的技术或算法；
★项为贝克链具有自主知识创新和核心竞争力的技术或算法 

# 6. 落地应用场景展望

## 6.1 轻节点——贝克钱包/BCOO PAY

目前，BCOO PAY由贝克链研发团队进行开发和运维，已经在IOS和安卓上线客户端。集成了密钥管理、智能支付、落地应用场景入口等多项功能。未来，贝克链主网上线后，BCOO PAY将会作为BCOO轻节点客户端迁移上链。

## 6.2 其他可能的应用及技术支持

贝克链为开发者创建友好的侧链和DApp开发环境。贝克基金会也将对重要的社交型应用提供技术支持，进行项目孵化，以帮助用户通过更多应用产品快速构建社交图谱和个人信誉。下表列出了某些可预见、可期待的应用产品，并对产品本身的技术需求和贝克链可提供的技术支持进行了分析和对比。


<table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" width="0" style="border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
 mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-border-insideh:.5pt solid windowtext">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:5.7pt">
  <td width="89" rowspan="2" style="width:66.4pt;border:solid #4F81BD 1.0pt;
  background:#2E74B5;mso-background-themecolor:accent1;mso-background-themeshade:
  191;padding:.75pt .75pt .75pt .75pt;height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="mso-bidi-font-size:12.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">应用类型</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:black"><o:p></o:p></span></b></p>
  </td>
  <td width="467" colspan="8" style="width:350.4pt;border-top:solid #4F81BD 1.0pt;
  border-left:none;border-bottom:solid white 1.0pt;border-right:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;mso-border-alt:solid #4F81BD 1.0pt;
  mso-border-bottom-alt:solid white .5pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:14.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">技术需求</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:black"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:5.7pt">
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">社交图谱贡献度</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">高吞吐</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">高并发</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">文件<span lang="EN-US"><o:p></o:p></span></span></b></p>
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">存储</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span class="GramE"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">低交易</span></b></span><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">成本</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">用户信息保护</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">商业信誉需求</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid white .5pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#2E74B5;mso-background-themecolor:
  accent1;mso-background-themeshade:191;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">频繁<span lang="EN-US"><o:p></o:p></span></span></b></p>
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><b style="mso-bidi-font-weight:normal"><span style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1;mso-font-kerning:0pt;mso-bidi-language:
  AR">迭代</span></b><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:white;mso-themecolor:background1"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">分布式聊天</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;color:gray;
  mso-themecolor:background1;mso-themeshade:128;mso-font-kerning:0pt;
  mso-bidi-language:AR">▲▲▲</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;
  color:gray;mso-themecolor:background1;mso-themeshade:128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">自媒体</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">▲▲</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">电商</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;color:gray;
  mso-themecolor:background1;mso-themeshade:128;mso-font-kerning:0pt;
  mso-bidi-language:AR">▲▲▲</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;
  color:gray;mso-themecolor:background1;mso-themeshade:128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">视频<span lang="EN-US">/</span>直播</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;mso-bidi-font-family:
  宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">▲</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">游戏</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">▲</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">区块链金融</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;color:gray;
  mso-themecolor:background1;mso-themeshade:128;mso-font-kerning:0pt;
  mso-bidi-language:AR">▲▲▲</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:&quot;Arial&quot;,sans-serif;mso-fareast-font-family:宋体;
  color:gray;mso-themecolor:background1;mso-themeshade:128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span class="GramE"><span style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black;
  mso-font-kerning:0pt;mso-bidi-language:AR">众包服务</span></span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">▲▲</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:white;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;mso-yfti-lastrow:yes;height:5.7pt">
  <td width="89" style="width:66.4pt;border:solid #4F81BD 1.0pt;border-top:none;
  mso-border-top-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="left" style="text-align:left;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:black;mso-font-kerning:0pt;
  mso-bidi-language:AR">共享经济</span><span lang="EN-US" style="font-size:11.0pt;
  line-height:150%;font-family:宋体;mso-bidi-font-family:宋体;color:black"><o:p></o:p></span></p>
  </td>
  <td width="68" style="width:51.3pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">▲▲</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.7pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span lang="EN-US" style="font-size:11.0pt;line-height:
  150%;font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★★★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.9pt;border-top:none;border-left:none;border-bottom:
  solid #4F81BD 1.0pt;border-right:solid #4F81BD 1.0pt;mso-border-top-alt:solid #4F81BD 1.0pt;
  mso-border-left-alt:solid #4F81BD 1.0pt;background:#B8CCE4;padding:.75pt .75pt .75pt .75pt;
  height:5.7pt">
  <p class="MsoNormal" align="center" style="text-align:center;mso-pagination:widow-orphan;
  vertical-align:middle"><span style="font-size:11.0pt;line-height:150%;
  font-family:宋体;mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;
  mso-themeshade:128;mso-font-kerning:0pt;mso-bidi-language:AR">★</span><span lang="EN-US" style="font-size:11.0pt;line-height:150%;font-family:宋体;
  mso-bidi-font-family:宋体;color:gray;mso-themecolor:background1;mso-themeshade:
  128"><o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

▲：代表该应用在贝克链PoR共识中可能的贡献程度。三角数量越多，贡献度越大。

★：代表该应用对某项需求的重要程度，五角星越多，需求性越高。

贝克链主网与侧链的交互关系如图4所示，大部分的商业应用可能会通过侧链运行，交易信息被记录在主网上，同时为主网提供信誉数据喂养。各个应用产品自带流量的同时，用户之间不可避免的存在交互共享（见图5），勾勒了更加丰富的社交图谱和生态系统。商业应用开发者由于本身具有更多元的社交图谱和频繁交易，往往可以累积更高的信誉值，被选为“诚信节点”的概率也越高，获得的系统奖励也越多。

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-6-2-1.png "图4. 贝克链主网与侧链的交互关系图")

图4. 贝克链主网与侧链的交互关系图

![alt text](https://raw.githubusercontent.com/BCOOCHAIN/BCOO/master/assets/Fig-6-2-2.png "图5. 各种应用之间可能存在的流量交互共享")


图5. 各种应用之间可能存在的流量交互共享

