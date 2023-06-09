# GA4FJSP
I used GA to solve FJSP problem records

## FJSP分类
FJSP分为两类，一类是T(Total)-FJSP，一类是P(Partial)-FJSP。<br>
两者的区别是：T-FJSP每个工序在所有机器上都有加工时间，而P-FJSP的每个工序在部分机器上有加工时间。<br>
T-FJSP是P-FJSP的一个特例。P-FJSP可以通过将没有加工时间的部分设置为很大的数值进而转换为T-FJSP。

## GA求解多工序FJSP
GA在求解多工序，并且工序之间存在并行工序的情况时。比如a,b两个产品a要先通过z工序，再通过x工序，b要先通过y工序，再通过x工序，这时就存在并行工序。<br>
即可以将不同工序的机器编码一起，生成一个包含所有机器的选择表，不通过的机器标记为不可选择。也可以将不同工序的机器单独编码，解码时单独处理。<br>
两种方式都有效，并且不会影响染色体长度，染色体长度由工序的多少决定。前者在code时好写一点，后者在逻辑上更好理解一些。

## GA实现交叉的几种方式：
### 一.
将机器选择部分（MS）和工序选择部分（OS）分别进行交叉操作，然后再将生成的两个新部分随机组合成新的染色体。<br>
这种实现方式在交叉操作中保留了MS和OS两个部分的所有信息，并且重新随机组合这些信息以生成新的染色体。<br>
这种方法可能会更全面地探索搜索空间，但也可能会引入不必要的噪声和冗余信息，从而降低算法的效率和性能。<br>
因此，选择合适的交叉操作方式需要考虑问题的具体特征，以及算法的性能和效率需求。

### 二.
染色体交叉只交叉MS或者OS，保留另一半。<br>
这种方式能够保留一个父代染色体中的有利特征，同时引入部分新的遗传信息，从而增加搜索空间的探索程度。<br>
然而，它们也可能会引入一些问题，例如可能会遗漏某些有利特征，或者可能会降低交叉操作的效率和性能。<br>
因此，需要进行实验和比较来确定这些方法是否能够提高算法的效率和性能。<br>

在小规模问题上目前来说两种方式都是有效的，但第二种方式貌似更快一点。大规模问题上的测试效果暂时未知。

## 针对机器故障的解决方案
### 右移

### 重调度

理论上重调度的效果好些。但重调度在处理大规模问题以及调度系统对反应时间敏感的情况下，不见得是个好选择。
