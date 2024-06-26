# CHEATING SHEET

## <u>说明</u>

本人总结了两份Cheating Sheet，一份用于机考，一份用于笔试。因为是自己使用，所以内容并不全面，主要是自己不太熟悉、容易疏忽或记不住的内容，且部分语言表述可能并不准确，因此从知识体系来看并不科学，但对自己来说，在两场考试中都起到了很大的作用。

## <u>机考CHEATING SHEET</u>

### 1.常用语法

```python
#eval()可将字符串形式的数字表达式或数字计算或提取出来(自动去除小数点后多余的0)

#format(x,'.yf')可将数字x保留y位小数

#round(x,y)可将数字x四舍六入(五看奇偶，奇入偶舍)，保留小数点后y位

#math库中的ceil函数和floor函数可分别向上取整和向下取整
#log10函数用法：y*log10(x)，log(x,y)函数计算以y为底的x的对数

#pow(x,y,z)==>x的y次幂模z所得余数

#lt_=lt.copy()可以复制集合、列表等

#from collections import Counter
a=['red', 'blue', 'red', 'green', 'blue', 'blue']
a=dict(Counter(a))==>{'blue': 3, 'red': 2, 'green': 1}
a.update(['blue', 'red', 'blue'])
print(a['blue'])    # 输出: 5
print(a.most_common(2))  # 输出 [('blue', 5), ('red', 3)]

#import bisect   ==>本质是二分查找
lst=[1,3,3,6,7,9]  #创建一个已排序的列表
index=bisect.bisect_left(lst,4) ==>index=3   #bisect_left函数用于查找给定元素应插入的索引
bisect.insort_left(lst,4) ==>lst=[1,3,3,4,6,7,9]  #insort_left函数用于插入给定元素并保持有序

#zfill(x)用于在字符串长度不足x时于左侧填充零字符，使字符串长度为x

#itertools的combinations组合函数C和permutations排列函数A,
设m为列表，n为数字，组合C=list(cb(m,n)) 排列A=list(pt(m))，生成以元组为元素的列表

#str模块：
1. str.lstrip(‘xyz...’) / str.rstrip(‘xyz...’)移除字符串左侧/右侧的字符’x’,’y’,’z’...(直到遍历到不需要删除的字符时停止)或移除空白字符
2. 在字符串中找子字符串首次出现的索引用str.find()，若不存在则返回-1，如果要找空格分开的独立字符串，如查找‘to’，为避免找到‘today’等非独立字符串，可写成(' '+string+' ').find(' '+to+' ')
3. str.startswith(x) / str.endswith(y) : 检查字符串是否以x开头或以y结尾
4.str.isalpha() / str.isdigit() / str.isalnum() : 检查字符串是否全部由字母/数字/字母和数字组成
5.将str中的前x个片段替换，没有x变量则全部替换：str=str.replace(piece,piece_,x)

#十进制数转二进制、八进制、十六进制:bin(num), oct(num), hex(num)
#二进制(八进制、十六进制)转十进制:int(str,2/8/16)(str为字符串类型的某进制数)

#x=sorted(x,key=lambda y:y[z])是通用排序方法,得到一个列表；若要对字典排序：dictionary=dict(sorted(dictionary.items(),key=lambda x:x[0/1]))

#列表反转lst.reverse()/lst=list(reversed(lst))/lst_=lst[::-1]，不要写成了lst.sort(reverse=True)
#字符串反转str_=str[::-1]

#将多个列表的对应元素合并为元组：
#tuple_list=list(zip(list_1,list_2,list_3......))

#为节省内存，求递推数列的列表可由关系项数个元素组成，不断向前递推
#例：对于a(n)=2*a(n-1)+a(n-2)数列，递推列表只需三个元素

#ASCII码表:(1)0-9对应48-57   (2)A-Z对应65-90   (3)a-z对应97-122
#查询ASCII值:ord('A')==>65    调用原符号:chr(65)==>'A'

#lst.sort(key=function)可通过定义函数function，实现根据映射值对原列表值排序
def function(num):
    return str(num)*10

#逐个输出列表中的元素(以空格间隔)：
print(*list)

#优先队列：
heappop(heap):返回并删除最小值
heapify(list):将list列表进行堆初始化
heappush(heap,item):将item添入堆中并重新排序
heappushpop(heap,item):将item放入堆中并返回heap最小值



for i,j in enumerate(list):
    lt.append([i,j])  ==>此时i为index，j为对应元素
#同时获得列表中相同元素的全部索引：
index=[i for i,value in enumerate(list) if value==X]

#测试组有n个元素，第i元素从列表lt[i]中选取一个值，构造所有可能的测试组ls：
#第一种写法：
for i in lt[0]:
    ls.append([i])
for i in range(1,n):
    ls_=ls[:]
    ls=[]
    for j in ls_:
        for k in lt[i]:
           ls.append(j+[k])
#第二种写法：
import itertools
lt1,lt2,lt3=[...],[...],[...]...
ls=itertools.product(lt1,lt2,lt3...)
ls=list(ls)

#位运算符(二进位):
(1)a&b:‘与’，对应位均为1，则结果为1，否则为0
(2)a|b:‘或’，对应位有一个为1，则结果为1，否则为0
(3)a^b:‘异或’，对应位不同时，结果为1，否则为0
(4)~a:‘取反’，对a的每个二进位取反，1变0，0变1
(5)a>>b:‘右移’，把a的每个二进位右移b位
(6)a<<b:‘左移’，把a的每个二进位左移b位，高位丢弃，低位补0

#双端队列:
from collections import deque
d = deque([])    创建一个deque队列
d.append(1)        从右端添加元素
d.appendleft(2)    从左端添加元素
x=d.pop()        从右端移除元素并返回
y=d.popleft()    从左端移除元素并返回

from functools import lru_cache
@lru_cache(None/num)  ==>None表示无限制缓存;num表示最大缓存递归次数
#Tips:使用lru_cache时，函数变量不能包含unhashable对象，即不可作为字典的键的对象，例如list，dict等类型

from collections import defaultdict
Dict=defaultdict(int)  ==>Dict[x]不存在时自动创建Dict[x]=0(默认值)
```



### 2.易错点

| **1:集合set()无序(不可排序)，不可索引，不可调用或遍历，不可把列表作为元素，可以把元组作为元素** |
| ------------------------------------------------------------ |
| **2:index()查找元素索引只能找到多个相同元素的第一个元素的索引** |
| **3:矩阵复制应当先把每个子列表复制，实现深拷贝**             |
| **4:若每组输入中有一空行，则input()；若要求每组输出以空行间隔，则print()** |
| **5:对列表、字符串等片段复制，超出范围的复制不会报错，无任何效果** |
| **6:一般递归的注意事项:**                                    |
| **6.1:若每个节点只能访问一次，意味着之前的分支访问过的节点将不能在之后的分支中访问** |
| **6.2:对于从所有结果中找出一种满足条件即可的题，使用flag作为全局变量，不能用return，因为return的结果将返还到上一次调用的函数，也就是仍在递归内部，无法一直返还到第一次调用的函数** |
| **6.3:递归写法一定要注意再次调用函数后变量的复原**           |
| **6.4:图搜索时注意起点和终点重合的情况、起点本身即成立/不成立的情况** |
| **6.5:列表只能是全局变量，对函数中列表的修改是全局修改**     |
| **7:大整数除法用/容易导致浮点数过大出错，须改用//**          |
| **8:对于整数的加减乘除运算，用round()替代int()以保证数值的准确性** |
| **9:爆栈时需import sys   sys.setrecursionlimit(一个大的阈值)** |
| **10:进行常见数据结构操作时若无效，则检查是否需要对变量重新赋值。如str=str.replace(x,y,z)操作需要重新赋值而lst.sort()操作不需要** |
| **11:处理字符串文本时提防标点符号和空格的特殊性**            |
| **12:对于结果很大，只需输出最后x位数字的题，直接取模即可**   |
| **13：用{}创建初始集合时{}内部可以有初始元素，而用set()创建集合时不可以有初始元素，只能为空集** |
| **14:考虑是否输入数据有重复**                                |





### 3.魔术方法

```python
#__init__(self,...):初始化方法，用于初始化对象的属性
#__str__(self):将对象转换为字符串时调用，常用于自定义对象的字符串表示
#__repr__(self):返回对象的“官方”字符串表示，通常可以通过eval(repr(obj))来恢复对象
#__len__(self):返回对象的长度，常用于自定义容器类
#__getitem__(self, key):获取指定键对应的值，常用于自定义容器类的索引操作
#__setitem__(self, key, value):设置指定键对应的值，常用于自定义容器类的索引赋值操作
#__add__(self, other):定义加法运算，当使用+运算符时调用
#__sub__(self, other):定义减法运算，当使用-运算符时调用

#如果self小于other，则返回True；否则返回False；如果无法确定大小关系，返回 NotImplemented
#__eq__(self, other): 定义相等性比较，当使用==运算符时调用
#__lt__(self, other): 定义小于比较。当使用<运算符时调用
#__ge__(self, other): 定义大于等于比较。当使用>=运算符时调用
#__ne__(self, other): 定义不等于比较。当使用!=运算符时调用
#__gt__(self, other): 定义大于比较。当使用>运算符时调用
#__le__(self, other): 定义小于等于比较。当使用<=运算符时调用
```



### 4.图论

```python
#图论算法总结：
#1.最短路径算法：
   #Dijkstra算法：用于找到两个顶点之间的最短路径。
   #Bellman-Ford算法：用于处理带有负权边的图的最短路径问题。
   #Floyd-Warshall算法：用于找到图中所有顶点之间的最短路径。
#2.最小生成树算法：
   #Prim算法：用于找到连接所有顶点的最小生成树。
   #Kruskal算法/并查集：用于找到连接所有顶点的最小生成树，适用于边集合已经给定的情况。
#3.拓扑排序算法：
   #DFS：用于对有向无环图（DAG）进行拓扑排序。
   #Karn算法/BFS ：用于对有向无环图进行拓扑排序。
#4.强连通分量算法：
   #Kosaraju算法/双DFS：用于找到有向图中的所有强连通分量。
   #Tarjan算法：用于找到有向图中的所有强连通分量。
   
#最小生成树(MST)
#最小生成树是包含无向连通图中所有顶点的树,具有最小的总权值,同时保证n-1个边权值中的最大值最小
#常见的建树算法有Prim算法和Kruskal算法：
#Prim算法：
#选择一个起始顶点作为树的根节点
#找到与当前树中一个顶点相邻的所有树外顶点中边权值最小的一个，将其加入树中。
#更新生成树与树外顶点的距离。
#重复以上步骤，直到所有顶点都被加入，最终得到的树即为最小生成树
from heapq import heappop,heappush
n=int(input())  #n个节点
dic={i:[] for i in range(n)}
for i in range(n):
    lt=list(map(int,input().split()))
    for l in range(n):
        dic[i].append((lt[l],l))  #每个节点相邻节点及边权值
nodes,ans=1,0
check=[False]+[True]*(n-1)
stack=[]
for i in dic[0]:
    heappush(stack,i)
while stack:
    length,node=heappop(stack)
    if check[node]:
        nodes+=1
        ans+=length
        check[node]=False
        if nodes==n:
            break
        for i in dic[node]:
            if check[i[1]]:
                heappush(stack,i)
print(ans)
#Kruskal算法/并查集：
#将所有边按照权值从小到大排序
#从权值最小的边开始遍历，如果这条边连接的两个顶点不在同一个连通分量中，则将其加入最小生成树中，
#并合并这两个连通分量
#重复上述步骤，直到最小生成树中包含了图中的所有顶点
#前置并查集代码class DisjointSet():
n=int(input())
def kruskal():
    Set=DisjointSet(n)
    MST=[]
    edges=[]
    for node in graph:
        for nbr,weight in graph[node]:
            edges.append((n,nbr,weight))
    edges.sort(key=lambda x:x[2])
    for edge in edges:
        u,v,weight=edge
        if Set.find(u)!=Set.find(v):
            Set.union(u,v)
            MST.append((u,v,weight))
    return MST
#Prim算法适用于稠密图，Kruskal算法适用于稀疏图

#判断有向图是否成环:拓扑排序
#拓扑排序是对有向无环图（DAG）排序的算法,包括Kahn算法和DFS(BFS也行，但效率更低)
#拓扑排序的结果是以顶点为元素的列表，任何有向路径都是从序列中索引小的顶点指向索引大的顶点
#常见的拓扑排序算法是Kahn算法:
#具体步骤:
#初始化：初始化一个用于存储结果序列的序列(可以有多个结果序列)和一个空集合（存储入度为0的顶点）
#计算入度：对于每个顶点，计算其入度，并将入度为0的顶点加入集合中
#循环处理：从集合中任意取出一个入度为0的顶点，并将其加入结果序列中，
#遍历以该顶点为起点的边，将这些边指向的顶点的入度减1。如果某个顶点的入度减为0，则将其加入集合中
#反复递归，直到集合为空
#检查环路：如果结果序列中包含了图中所有的顶点，则拓扑排序成功；否则说明图中存在环路，无法拓扑排序
n,m=map(int,input().split())
dic={i:[] for i in range(1,n+1)}  #存储以各节点为出发点可以指向的nbrs
lt=[0]*n   #存储各节点入度
for _ in range(m):
    x,y=map(int,input().split())
    dic[x].append(y)
    lt[y-1]+=1
ans,stack=[],[]
for i in range(n):
    if lt[i]==0:
        stack.append(i+1)
while stack:
    cur=stack.pop()  #此处可以任意取出一个入度为0的顶点，若要求字典序最小，则使用heapq
    ans.append(cur)
    for nbr in dic[cur]:
        lt[nbr-1]-=1
        if lt[nbr-1]==0:
            stack.append(nbr)
if len(ans)==n:
    print('拓扑排序成功')
else:
    print('存在环路，无法拓扑排序')
        
#判断无向图是否连通:
check=[False]+[True]*(n-1)
def if_connected(n,node):
    for nbr in dic[node]:
        if check[nbr]:
            check[nbr]=False
            if_connected(n,nbr)
if_connected(n,0)

#判断无向图是否成环:以下是BFS代码，还可以使用DFS或并查集方法
check=[True]*n
def if_loop(n):
    for node in range(n):
        if check[node]:
            check[node]=False
            stack=[(node,[node])]
            while stack:
                cur,vis=stack.pop()
                for nbr in dic[cur]:
                    if nbr in vis[:-2]:
                        return True
                    if nbr not in vis:
                        check[nbr]=False
                        stack.append((nbr,vis+[nbr]))
    return False

#Dijkstra算法：用于解决单源最短路径问题。基本思想是通过不断扩展离源节点最近的节点来逐步确定最短路径。
#具体步骤：(设节点数为n)
#初始化长度为n的数组，用于记录到所有节点的最短路径长，设到源节点的距离为0，其他节点的距离为无穷大。
#选择一个未访问的节点中距离最小的节点作为当前节点。
#遍历当前节点的邻居节点，如果通过当前节点到达邻居节点的路径比已知最短路径更短，则更新相应的最短路径。
#遍历完所有邻居节点后，标记当前节点为已访问。
#重复上述步骤，直到所有节点都被访问或者所有节点的最短路径都被确定。
from heapq import heappush,heappop
k,n,r=int(input()),int(input()),int(input())
dic={i:[] for i in range(1,n+1)}
for _ in range(r):
    s,d,l=map(int,input().split())
    dic[s].append((l,d))
stack=[]
heappush(stack,(0,1))
def dijikstra():
    while stack:
        l,d=heappop(stack)
        if d==n:
            return l
        for nbr in dic[d]:
            l_,d_=l+nbr[0],nbr[2]
            heappush(stack,(l_,d_))
    return -1
print(dijikstra())

#Bellman-Ford算法：用于解决单源最短路径问题，可以处理负权边。
#基本思想是通过松弛操作逐步更新节点的最短路径估计值，直到收敛到最终结果。
#具体步骤：(设节点数为n，边数为E)
#初始化长度为n的数组，用于记录到所有节点的最短路径长，设到源节点的距离为0，到其他节点的距离为无穷大。
#进行n-1次循环，每次循环对E条边(u,v)依次进行松弛操作:
#如果当前已知的从源节点到节点u的最短路径+边(u, v)的权重之和
#比当前已知的从源节点到节点v的最短路径更短，则更新相应最短路径。
#如果在n-1次循环后，仍然可以通过松弛操作更新最短路径，则说明存在负权回路，因此无法确定最短路径。

#Floyd-Warshall算法，用于解决多源最短路径问题。
#Floyd-Warshall算法可以在有向图或无向图中找到任意两个顶点之间的最短路径。
#基本思想是用二维数组存储任意两个顶点之间的最短距离。
#初始时，这个数组包含图中各对单边连通的顶点之间的单边权重，对于不单边相连的顶点，权重为无穷大。
#迭代更新数组，逐步求得所有顶点之间的最短路径。
#具体步骤：
#初始化一个二维数组dist，用于存储任意两个顶点之间的最短距离。
#初始时，dist[i][j]表示顶点i到顶点j的直接边的权重，如果i和j不直接相连，则权重为无穷大。
#对于每个顶点k，考虑顶点k作为中间节点。
#遍历所有的顶点对(i, j),如果通过顶点k可以使得从顶点i到顶点j的路径变短，则更新dist[i][j]为更小的值。
#dist[i][j]=min(dist[i][j], dist[i][k]+dist[k][j])
#最终，dist数组中存储的就是所有顶点之间的最短路径。
#适用于稠密图的最短路径问题，并且可以处理负权边和负权回路。
def floyd_warshall(graph):
    n=len(graph)
    dist=[[float('inf')]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i][j]=0
            elif j in graph[i]:
                dist[i][j]=graph[i][j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j])
    return dist

#骑士周游:DFS，Warnsdorff算法用于解决是否能遍历所有顶点且恰好只经过一次的问题
#主要思想：不要遍历所有的下一步可到达顶点，而是直接选择下一步可到达顶点中可选路径最少的那个顶点

#词梯问题:建桶，然后BFS
bucs,words={},{}
for _ in range(int(input())):
    word=input()
    lt=[word[:3]+'_','_'+word[1:],word[:2]+'_'+word[3],word[0]+'_'+word[2:]]
    for buc in lt:
        bucs[buc]=bucs.get(buc,[])+[word]
        words[word]=words.get(word,[])+[buc]
        
#Kosaraju算法是一种用于在有向图中寻找强连通分量的算法，核心思想就是两次DFS。
#步骤如下:
#1. 第一次DFS：对图进行标准的DFS，在此过程中记录下顶点完成搜索的顺序。
#这一步的目的是为了找出每个顶点的完成时间。
#2. 反向图：接下来，我们对原图取反，将所有的边方向反转，得到反向图。
#3. 第二次DFS：在第二次DFS中，我们按照第一步中记录的顶点完成时间的逆序，
#对反向图进行DFS。找出反向图中的强连通分量。
def dfs1(graph, node, visited, stack):
    visited[node]=True
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs1(graph, nbr, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, ans):
    visited[node]=True
    ans.append(node)
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs2(graph, nbr, visited, ans)

def kosaraju(graph):
    stack=[]
    visited=[False]*n
    for node in range(n):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    tra_graph=[[] for _ in range(n)]
    for node in range(n):
        for nbr in graph[node]:
            tra_graph[nbr].append(node)
    visited=[False]*n
    answers=[]
    while stack:
        node=stack.pop()
        if not visited[node]:
            ans=[]
            dfs2(tra_graph, node, visited, ans)
            answers.append(ans)
    return answers
```



### 5.线性表

```python
#单调栈
n=int(input())
lt=list(map(int,input().split()))
ans=[0]*n  #用于存储第i个元素之后第一个大于lt[i]的第j个元素的位置j，若不存在，则记为0
stack=[(lt[-1],n)]
for i in range(n-2,-1,-1):
    while stack and lt[i]>=stack[-1][0]:
        stack.pop()
    if stack:
        ans[i]=stack[-1][1]
    else:
        ans[i]=0
    stack.append((lt[i],i+1))
print(*ans)
#数据较大时为避免MLE：
#1.改进输出方式，开始直接建元素全为0的ans列表，而不要建空的ans列表，那样相当于多设一个列表
#2.不能一开始给所有元素配上索引，这样需要多设置一个列表，可以改成在遍历的过程中给栈中每个元素配上索引
 
#十进制转换为其他进制
def base_converter(num_10,base):
    stack=[]
    digits='0123456789ABCDEF'
    while num_10>0:
        stack.append(num_10%base)
        num_10//=base
    num_base=''
    while stack:
        num_base+=digits[stack.pop()]
    return num_base

#中序表达式转前/后序表达式
#Shunting Yard(调度场算法)是一种将中缀表达式转换为前/后缀表达式的算法
#主要思想是使用两个栈（运算符栈和输出栈）来处理表达式的符号
#按照运算符的优先级和结合性，将符号逐个处理并放置到正确的位置

#中缀表达式转后缀表达式的具体步骤:
#1.建立空运算符栈和输出栈
#2.从左到右遍历中缀表达式的每个符号
#    ①如果是操作数（数字/字母）:将其添加到输出栈
#    ②如果是左括号:将其推入运算符栈
#    ③如果是运算符:
#        如果该运算符的优先级大于运算符栈顶的运算符，或者运算符栈顶是左括号，将该运算符推入运算符栈
#        否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满足上述条件（或者运算符栈为空）
#        再将当前运算符推入运算符栈
#    ④如果是右括号:将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号，将左括号弹出。
#3.遍历完成后如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中

#Tip:将中缀表达式转换为前缀表达式的算法同理，只需将具体步骤中的"左"和"右"互换
for _ in range(int(input())):
    string=input()
    priority={'+':1,'-':1,'*':2,'/':2}
    output,calcu=[],[]
    num=''    #num用于存储每一个完整的数字
    for i in string:
        if 48<=ord(i)<=57 or ord(i)==46:  #'0123456789'或'.'
            num+=i
        else:
            if num!='':
                output.append(num)
            num=''
            if ord(i)==40:  #'('
                calcu.append(i)
            elif ord(i)==41:  #')'
                while calcu[-1]!='(':
                    j=calcu.pop()
                    output.append(j)
                calcu.pop()
            else:  #'+-*/'
                while calcu!=[] and calcu[-1]!='(' and priority[i]<=priority[calcu[-1]]:
                    j=calcu.pop()
                    output.append(j)
                calcu.append(i)
    if num!='':
        output.append(num)   #以小数结尾的表达式需要补录最后一个小数
    while calcu!=[]:
        j=calcu.pop()
        output.append(j)
    ans=' '.join(output)
    print(ans)
    
#后序/前序表达式求值(以后序表达式为例)
s=list(input().split())
stack=[]
num=''
for i in s:
    if 48<=ord(i)<=57 or ord(i)==46:
        num+=i
    else:
        if num!='':
            stack.append(num)
            num=''
        a,b=stack.pop(),stack.pop()
        c=str(eval(a+i+b))
        stack.append(c)
print(format(stack[0],'.2f'))

#合法出栈序列
s1,s2=list(input()),list(input())
l1,l2=len(s1),len(s2)
if l2!=l1:
    print('NO')
else:
    stack=[]
    j=-1
    for i in range(l1):
        while stack==[] or s2[i]!=stack[-1]:
            j+=1
            if j<l1:
                stack.append(s1[j])
            else:
                break
        if stack and s2[i]==stack[-1]:
            stack.pop()
    if i==l1-1 and stack==[]:
        print('YES')
    else:
        print('NO')
        
#heap堆+dic字典求动态数组最大/小值(以滑动窗口为例)
from heapq import heappush,heappop
n,k=map(int,input().split())
lt=list(map(int,input().split()))
ls=[]
for i in lt:
    ls.append(-i)
dic,stack,ans={},[],[]
for i in range(k):
    dic[ls[i]]=dic.get(ls[i],0)+1
    heappush(stack,ls[i])
ini=heappop(stack)
ans.append(-ini)
heappush(stack,ini)
for j in range(n-k):
    dic[ls[j]]-=1
    dic[ls[j+k]]=dic.get(ls[j+k],0)+1
    heappush(stack,ls[j+k])
    cur=heappop(stack)
    while dic[cur]<1:
        cur=heappop(stack)
    ans.append(-cur)
    heappush(stack,cur)
print(*ans)

#用Stack迭代实现DFS(以n皇后为例)
#基本思路是在定义的函数中建立初始栈，只有一个初始元素，然后while Stack:循环，
#每次循环pop()出最右端的元素，若其满足条件则输出，若不满足则遍历寻找所有与其相关联的
#下一步元素，依次append到栈中
        
#bisect二分法求最长严格单调下降子序列的长度
import bisect
def longest_decrease_sub(nums):
    nums=list(reversed(nums))
    ans=[]  
    for num in nums:
        index=bisect.bisect_left(ans,num)
        if index==len(ans):
            ans.append(num)
        else:
            ans[index]=num
    return len(ans)
n=int(input())
nums=list(map(int,input().split()))
print(longest_decrease_sub(nums))  
#存在一个原序列的最长严格单调下降子序列，使得以该序列的每个数为终点，对应一个最长单调上升子序列
#将这些子序列合并起来恰好是原序列
#基于上述Dilworth定理，可以通过求出原序列最长严格单调下降子序列的长度的方法得到答案
#ans序列不一定是原序列的最长严格单调下降子序列，但是ans序列长度始终后者的长度

#括号嵌套检测
s=list(input())
def isValid(s):
    stack=[]
    flag=False
    mapp={')':'(','}':'{',']':'['}
    for i in s:
        if i in mapp.values():
            if stack!=[]:
                flag=True
            stack.append(i)
        elif i in mapp.keys():
            if stack==[]:
                return 'Error'
            elif mapp[i]!=stack.pop():
                return 'Error'
    if stack!=[]:
        return 'Error'  #不合法
    elif flag:
        return 'Yes'  #合法，且存在括号嵌套
    return 'No'  #合法，且不存在括号嵌套
```



### 6.树

```python
#1.前、中、后序遍历二叉树
    def traversal(self,order):    
        if order=='preorder':
            print(self.key,end='')
        if self.left!=None:
            self.left.traversal(order)
        if order=='midorder':
            print(self.key,end='')
        if self.right!=None:
            self.right.traversal(order)
        if order=='postorder':
            print(self.key,end='')

#2.二叉树的创建:括号嵌套
def Bulidtree(test):
    stack=[]
    cur=Node('0')   #该节点为伪根节点，实际根节点是其左子节点，实际二叉树是其左子树
    for j in test[1:]:
        if j=='(':
            stack.append(cur)
        elif j==')':
            node=stack.pop()
            if node.val!='0':
                cur=node   #保证最后返还的cur是实际根节点
        else:
            cur=Node(j)
            if stack[-1].left==None:
                stack[-1].left=cur
            else:
                stack[-1].right=cur
    return cur

#3.解析树的创建:
   #将中序表达式转换为二叉解析树
#前置类
#class Stack
#class Binarytree
def parse_tree_mid(exp):
    lt=exp.split()
    stack=Stack()
    root=Binarytree('')
    stack.push(root)
    current=root
    for i in lt:
        if i=='(':
            stack.push(current)
            current.InsertLeft('')
            current=current.left
        elif i==')':
            current=stack.pop()
        elif i in '+-*/':
            current.SetRootVal(i)
            current.InsertRight('')
            stack.push(current)
            current=current.right
        else:
            current.SetRootVal(int(i))
            father=stack.pop()
            current=father
    return root
#Tips:该解析树是根据中序表达式建立的，对其进行前、后序遍历得到的结果不等于对应的前、后序表达式
  #中序遍历还原中序表达式
def print_mid_parse_tree(root):
    piece=''
    if root:
        piece+='('+print_mid_parse_tree(root.left)
        piece+=root.val
        piece+=print_mid_parse_tree(root.right)+')'
    return piece

#计算二叉解析树
import operator
def evaluate(node):
    operators={'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv}
    l_node=node.left
    r_node=node.right
    if l_node and r_node:
        oper=operators[node.val]
        return oper(evaluate(l_node),evaluate(r_node))
    else:
        return node.val

#4.将后序表达式转换为二叉解析树
def parse_tree_post(exp):
    stack=[]
    for i in exp:
        current=Node(i)
        if i in '+-*/':
            current.right=stack.pop()
            current.left=stack.pop()
        stack.append(current)
    return stack[0]
  #将前序表达式转换为二叉解析树:exp=list(reversed(exp)),其余代码同上
  #按层次遍历二叉解析树的结果，再将其反转就得到队列表达式
#根据二叉树前中序遍历求后序遍历
def post(pre,mid):
    if pre==[]:
        return ''
    Index=mid.index(pre[0])
    l_mid,r_mid=mid[:Index],mid[Index+1:]
    l_pre,r_pre=[],[]
    for i in pre[1:]:
        if mid.index(i)<Index:
            l_pre.append(i)
        else:
            r_pre.append(i)
    return post(l_pre,l_mid)+post(r_pre,r_mid)+pre[0]
#根据二叉树中后序遍历求前序遍历
def pre(mid,post):
    if mid==[]:
        return ''
    Index=mid.index(post[-1])
    l_mid,r_mid=mid[:Index],mid[Index+1:]
    l_post,r_post=[],[]
    for i in post[:-1]:
        if mid.index(i)<Index:
            l_post.append(i)
        else:
            r_post.append(i)
    return post[-1]+pre(l_mid,l_post)+pre(r_mid,r_post)
#Tips:根据二叉树前后序遍历无法得到确定的中序遍历

#5.哈夫曼编码树的构建
  #哈夫曼编码树的叶子节点均为字符节点，其余节点均为None
  #从根节点开始，向左走为0，向右走为1，每个字符节点的路径即为其哈夫曼编码
import heapq

class Node:
    def __init__(self,char,fre):
        self.char=char
        self.fre=fre
        self.left=None
        self.right=None
    def __lt__(self,other):
        return self.fre<other.fre
    #若要求同频率节点需比较字符集:
    #if self.fre!=other.fre:
    #    return self.fre<other.fre
    #else:
    #    return self.char<other.char

def huffman_codeing(char_fre):
    heap=[Node(char,fre) for char,fre in char_fre.items()]
    heapq.heapify(heap)
    while len(heap)>1:
        left=heapq.heappop(heap)
        right=heapq.heappop(heap)
        merge=Node(None,left.fre+right.fre) 
        merge.left=left                    
        merge.right=right
        heapq.heappush(heap,merge)
    return heap[0]
  #若要求同频率节点需比较字符集，则节点char均为列表，merge节点char设为left.char+right.char
  
  #计算带权外部路径长度
def length(node,depth):  
    if node.left==node.right==None:
        return depth*node.fre
    return length(node.left,depth+1)+length(node.right,depth+1)

  #构建哈夫曼编码字典
def Code(node,code):
    if node.left==node.right==None:
        encodes[node.char[0]]=code
        decodes[code]=node.char[0] #叶子节点char列表只有1个字符，需要提取出来
        return
    Code(node.left,code+'0')
    Code(node.right,code+'1')

char_fre={input()}
huffman_tree=huffman_codeing(char_fre)
encodes,decodes={},{}
Code(huffman_tree,'')

#6.二叉搜索树(二叉排序树/二叉查找树/BST树)具有二叉搜索性:
  #性质:小于父节点的节点都在左子树中，大于父节点的节点都在右子树中
  #节点的删除：
     #叶子节点：直接删除
     #只有一个子节点的节点：用其子节点替代，然后删除原先的子节点
     #有两个子节点的节点：找到该节点的后继节点（比待删除节点大的最小节点,即右子树中的最小节点）
     #或前驱节点（比待删除节点小的最大节点,即左子树中的最大节点）
     #来替代它，然后删除原先的后继节点或前驱节点
     #具体选择标准是让删除节点后的二叉树尽量平衡
#二叉搜索树按层次遍历:
def insert(root,val):
    if root==None:
        return Node(val)
    if val<root.val:
        root.left=insert(root.left,val)
    else:
        root.right=insert(root.right,val)
    return root

def level_order_traversal(root):
    queue=[root]
    traversal=[]
    while queue:
        current=queue.pop(0)  #也可用双端队列popleft
        traversal.append(current.value)
        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)
    return traversal

nums=[input()]
root=None
for num in nums:
    root=insert(root,num)
traversal=level_order_traversal(root)

#7.平衡二叉树(AVL树)==>能自动维持平衡的二叉搜索树
#树型判断:LR型指的是平衡因子为2、-1的情形，其他情况同理
#为了保持AVL树的平衡,插入或删除节点时可能需要进行左旋、右旋、左右旋和右左旋
#具体地，LL型右旋，RR型左旋，LR型先左旋后右旋，RL型先右旋后左旋
  #左(右)倾二叉树的最坏情况是给定层数实现最少节点和给定节点数实现最多层数的情形
  #n层的AVL树至少有m个节点:递推公式m(n)=m(n-1)+m(n-2)+1
  #平衡二叉树的建立
class Node:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
        self.height=1  #存储节点的高度，以便计算节点的平衡因子
        
def Height(node):
    if node:
        return node.height
    return 0
    
def Balance(node):
    if node:
        return Height(node.left)-Height(node.right)
    return 0
 
def insert(root,insert_node):
    if root==None:
        return insert_node
    else:
        if root.val>insert_node.val:
            root.left=insert(root.left,insert_node)
        else:
            root.right=insert(root.right,insert_node)
        root.height=1+max(Height(root.right),Height(root.left))  #节点高度通用计算式
        balance=Balance(root)
        if balance==2:
            if insert_node.val>root.left.val:  
                root.left=left_rotate(root.left,root.left.right)  #更新根节点的左子节点
            return right_rotate(root,root.left)  #返还旋转后的局部根节点
        #若插入节点在root.left的右子树中，则root.left的平衡因子为-1
        elif balance==-2:
            if insert_node.val<root.right.val:
                root.right=right_rotate(root.right,root.right.left)
            return left_rotate(root,root.right)
        return root
    
def right_rotate(node_A,node_B):
    node_A.left=node_B.right
    node_B.right=node_A
    node_A.height=1+max(Height(node_A.left),Height(node_A.right))
    node_B.height=1+max(Height(node_B.left),Height(node_B.right))
    return node_B  #旋转后局部根节点变成node_B

def left_rotate(node_A,node_B):
    node_A.right=node_B.left
    node_B.left=node_A
    node_A.height=1+max(Height(node_A.left),Height(node_A.right))
    node_B.height=1+max(Height(node_B.left),Height(node_B.right))
    return node_B

def pre_order(root):
    if root:
        return [str(root.val)]+pre_order(root.left)+pre_order(root.right)
    else:
        return []
    
n=int(input())
nums=map(int,input().split())
root=None
for num in nums:
    insert_node=Node(num)
    root=insert(root,insert_node)
ans=pre_order(root)
print(' '.join(ans))

#8.并查集(Disjoint Set)
#初始的n个节点均为叶子节点，每次union相当于将两个节点统一为其中一个节点的左右子节点
#从而每一棵二叉树的根节点可视为整个二叉树的代表节点
#Tips:self.parent列表中不同节点的数量不等于根节点数量，需要遍历每个节点，
#追溯其对应的根节点，来计算根节点总数
class DisjointSet:
    def __init__(self,n):
        self.parent=[i for i in range(n)]
        self.rank=[0]*n
        self.size=[1]*n
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union_by_rank(self,x,y):
        root_x,root_y=self.find(x),self.find(y)
        if root_x!=root_y:
            if self.rank[root_x]>self.rank[root_y]:
                self.parent[root_y]=root_x
            elif self.rank[root_x]<self.rank[root_y]:
                self.parent[root_x]=root_y
            else:
                self.parent[root_y]=root_x
                self.rank[root_x]+=1
    def union_by_size(self,x,y):
        root_x,root_y=self.find(x),self.find(y)
        if root_x!=root_y:
            if self.size[root_x]>self.size[root_y]:
                self.parent[root_y]=root_x
                self.size[root_x]+=self.size[root_y]
            else:
                self.parent[root_x]=root_y
                self.size[root_y]+=self.size[root_x]
#对于已知整体分组的数量为m组，题目每次只告知2个节点间的同组或不同组的信息，需要实时更新已知
#所有分组信息并做出判断。
#此类题目的Disjointset初始化如下:
#   def __init__(self,n):
#       self.parent=[i for i in range(m*n+1)]
#       self.rank=[0]*(m*n+1)
#对于每次告知的2个节点间的信息，需要进行m对union,比如说x，y同组的union情况如下:
#Set.union(x,y)
#Set.union(x+n,y+n)
#Set.union(x+2*n,y+2*n)
#......
#Set.union(x+(m-1)*n,y+(m-1)*n)
#对于每次需要判断的2个节点间的关系，需要查找root_x,root_y,root_y+n,...root_y+(m-1)*n
#相应的有root_x==root_y;root_x==root_y+n,...root_x==root_y+(m-1)*n共n种关系可能

#9.给出伪满二叉树(用#等符号代替叶子节点的空子节点)的前序遍历，要求建树:
i=0
lt=list(input().split())
def Buildtree():
    global i
    if lt[i]!='#':
        cur=Node(lt[i])
        i+=1
        cur.left=Buildtree()
        cur.right=Buildtree()
        return cur
    else:
        i+=1
        return None
root=Buildtree()
#判别所给lt是否是合法的前序遍历:
    #出度入度判别法，一个非#节点有2个出度，所给测试数据必须满足所有非#节点都恰好有2个出度
if lt[0]=='#':
    print('False')
else:
    stack=[lt[0],lt[0]]
    flag=True
    for i in lt[1:]:
        if stack:
            stack.pop()
            if i!='#':
                stack+=[i,i]
        else:
            flag=False
            break
    if flag and stack==[]:
        print('True')
    else:
        print('False')

#10.字典树判断前缀是否一致
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
class Trie:
    def __init__(self):
        self.root = TrieNode()
	def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
				node = node.children[char]
		node.is_end_of_word = True
	def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
			node = node.children[char]
		return node.is_end_of_word
	def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
			node = node.children[char]
		return True
#法二:
for _ in range(int(input())):
    n=int(input())
    nums=[input() for _ in range(n)]
    nums.sort()
    def Judge():
        for i in range(n-1):
            if nums[i+1].startswith(nums[i]):
                return 'NO'
        return 'YES'
    ans=Judge()
    print(ans)
#电话号码按字典序排序后，具有共同前缀的电话号码会相邻排列，因此只需检查相邻的电话号码即可判断是否一致
```



### 7.计概内容

```python
#冒泡排序思维的推广：适用于通过多次对相邻2个元素的操作来实现整体成立的题型

#DP(动态规划：通过维护数/列表以保证局部最优)
#单列表一维DP:
#关键在于dp[i]时的最优结果判断，通过状态转移方程实现，max(A,B,C......)中的值一般为
#dp[i-x]或dp[i-x]+lst[i-y]或dp[i-x]+number
#矩阵二维DP:
#行为每次操作(n+1行)，列为限定的量(m+1列)
#双重遍历，在状态转移方程中的比较值表达式存在行间关联和
#列间关联，形式上仍为dp[i-x] [j-y]或dp[i-x] [j-y]与lst[l-p] [k-q]的组合
#双列表一维DP：思考出发点在于R[i]不一定由R[i-1]独立确定,类似于0,1变换的反转操作
#意味着n时R[i]需要同时考虑n-1时R[i-1]和B[i-1]，故建立双列表一维DP

#例题一：最长上升子序列(一维DP)
#遍历原序列for i in lst，依次以i为递增子序列的最终项，ans[i]初始为1，从i之前的ans中
#找出小于i且最接近i的j对应的子序列ans[j]+1即为ans[i]

#例题二：田忌赛马(一维DP)
#以一对马为比较单位，若A(max)>B(max),则以此赢一次，若A(max)<B(max)，则用A(min)
#和B(max)输一次，若A(max)==B(max),则看A(min)和B(min)，若A(min)<B(min),则用A(min)
#和B(max)输一次，若A(min)>B(min),则以此赢一次，若A(min)==B(min),则用A(min)和B(max)比

#例题三：采药(0-1背包的最优解问题)(二维DP)
#构建矩阵，行为零行和n种草药，列为从第0秒到第m秒
#进行for _ in range(m+1)和for _ in range(n+1)的双重循环，
#先判断时间是否允许采这种药，再判断(采这种药的收益＋上一行中扣除该次
#采药所需时间后对应的收益)与上一行中同一时间对应收益的大小关系，取其中的大值

#例题四：红蓝玫瑰(双列表一维DP)
#if ls[i]==0:
#    R[i]=min(R[i-1],B[i-1]+1)
#    B[i]=min(B[i-1]+1,R[i-1]+1)
#else:
#    R[i]=min(R[i-1]+1,B[i-1]+1)
#    B[i]=min(B[i-1],R[i-1]+1)

#例题五：NBA门票(多重背包的最优解问题)(二维DP)(背包指的是物品的数量，故采药题是0-1背包)
nums=list(map(int,input().split()))
prices=[1,2,5,10,20,50,100]
dp=[0]+[float('inf')]*(n)
for i in range(7):
    price=prices[i]
    num=nums[i]
    k=1
    while num>0:
        num_use=min(num,k)
        num-=num_use
        for j in range(n,num_use*price-1,-1):
            dp[j]=min(dp[j],dp[j-num_use*price]+num_use)
        k*=2
#基本思路：
#二维DP可以用一个列表反复更新实现，列表除第一个元素取0外，其他初始元素全部取float('inf')(即无穷大)，即dp=[float('inf')]*(n+1)
#行：门票价位[1,2,5,10,20,50,100]   列：奖金[0,1,2...n]
#先遍历门票价位，再遍历该价位门票的数量，再(重点)按倒序遍历奖金，反复更新相应奖金对应的最少票数==>如果门票数量任意，则直接顺序遍历，无需二进制优化
#简化方法：二进制拆分
#将某一价位门票的数量表示成二进制，如34=1+2+4+8+16+3，这么做只需要相应遍历6次奖金，而不需要遍历34次

#例题六：神奇的口袋(0-1背包的方案数问题)
dp=[1]+[0]*40
for i in lt:
    for j in range(40,i-1,-1):
        dp[j]=dp[j-i]+dp[j]
print(dp[-1])
#基本思路：先遍历物品体积lt[i]，再(重点)按倒序遍历容量j，使用一个列表dp[j]反复更新

#例题七：Coins(多重背包的方案数问题)
dp=1
mask=(1<<(m+1))-1
#mask作为掩码，用于保证dp值在[1,2**(m+1)-1]之间，bin(mask)==>0b11...1(共(m+1)个1)
for i in range(n):
    coin=a[i]
    num=c[i]
    k=1
    while num>0:
        use=min(k,num)
        num-=use
        dp=(dp|(dp<<(use*coin)))&mask
        k*=2
print(bin(dp).count('1')-1)
#最后用二进制表示dp值，数出'1'的个数再减1
#基本思路：用0-1表示每一金额有无相应方案，先遍历硬币面值，再使用二进制优化按倒序遍历金额，只要dp[j-use*coin]|dp[j]==1,则dp[j]=1
#优化方法：位运算代替列表dp

#二分查找：
#left,right=num1,num2
#middle=(left+right)/2
#while right-left>(一个较小的数):
#    (判断middle是否成立，并相应地将left=middle或right=middle)
#    middle=(left+right)/2

#任给一个区间，找最小覆盖所需覆盖体个数的问题:
#类似冒泡排序，多次遍历，一轮结束之后从左往右第一个没被盖住的是点x,
#则下一轮扫能盖住点x的覆盖体，在所有能盖住x的覆盖体中选择右端点最靠右的，
#然后更新从左往右第一个没被盖住的点x，直到所有点都能被盖住

#"延迟操作"：(以expedition为例题)每遍历一座加油站，若此时p<0，则从heap堆中挑选已遍历的加油站中油量最大的加油，直到p>=0，就将新加油站的油量存入heap堆，继续往下遍历；或heap堆成为空集(即无法到达城镇)，结束遍历
flag=False
for i in lt:
    p-=(l-i[0])
    l=i[0]
    while p<0:
        if heap==[]:
            flag=True
            break
        p-=heappop(heap)
    heappush(heap,-i[1])
    if flag:
        print(-1)
        break
if flag==False:
    print(n+1-len(heap))

#区间问题：将区间排序（通常在从左往右扫的时候按右端点排序），按顺序遍历每一个区间，选择第一个满足要求的区间，同时更新剩下区间所需满足的要求（通常是区间右端点），以此类推
#例题：畜栏保留问题：按开始时间进行排序，保证第一个畜栏开始占用的时间是最早的ans队列用于按输入顺序记录相应结果。然后遍历排序后的队列，若开始时间大于结束时间最早的现有畜栏，则进入该畜栏；否则新增一个畜栏
from heapq import heappop,heappush
n=int(input())
lt=[]
for i in range(n):
    lt.append(list(map(int,input().split()))+[i])
lt.sort()
ans=[0]*n
ans[lt[0][2]]=1
heap=[[lt[0][1],lt[0][0],lt[0][2]]]
num=1
for i in range(1,n):
    ls=heappop(heap)
    if lt[i][0]>ls[0]:
        ans[lt[i][2]]=ans[ls[2]]
    else:
        num+=1
        ans[lt[i][2]]=num
        heappush(heap,ls)
    heappush(heap,[lt[i][1],lt[i][0],lt[i][2]])
print(num)
for _ in ans:
    print(_)
    
#整数划分问题(本质是二维DP问题)
#①将n划分成若干正整数之和
DP=[[1]*(n+1)]+[[0]*(n+1) for i in range(n)]
for i in range(1,n+1):
   for j in range(1,n+1):
      if i==j:
        DP[i][j]=DP[i][j-1]+1 
      elif i<j:
        DP[i][j]=DP[i][i]
      elif i>j:
        DP[i][j]=DP[i][j-1]+DP[i-j][j]
print(DP[-1][-1])

#②将n划分成k个正整数之和
dp=[[0]*(k+1) for i in range(n+1)]
for i in range(n+1):
   dp[i][1]=1
for i in range(1,n+1):
   for j in range(1,k+1):
      if i>=j:
         dp[i][j]=dp[i-1][j-1]+dp[i-j][j]
print(dp[-1][-1])
#dp[i][j]的所有划分可以分为有1的划分和没有1的划分，前者与dp[i-1][j-1]一一对应，后者与dp[i-j][j]一一对应

#③将n划分成若干不同的正整数之和
Dp=[[0]*(n+1) for i in range(n+1)]
for i in range(1,n+1):
   for j in range(1,n+1):
      if i==j:
         Dp[i][j]=Dp[i][j-1]+1
      elif i<j:
         Dp[i][j]=Dp[i][i]
      elif i>j:
         Dp[i][j]=Dp[i][j-1]+Dp[i-j][j-1]
print(Dp[-1][-1])
#Dp[i][j]表示将i划分成最大数不大于j的划分数，当i>j时，Dp[i][j]包括Dp[i][j-1]和最大数为j，剩余i-j的划分中最大数不大于j-1的划分数，即Dp[i-j][j-1]

#④将n划分成若干奇正整数之和
dP=[[0]*(n+1) for i in range(n+1)]
for i in range(n+1):
   dP[i][1]=1
for i in range(1,n+1):
   for j in range(1,n+1,2):    #只对j为奇数的元素作dp
      if i==j and j>1:
         dP[i][j]=dP[i][j-2]+1 
      elif i<j:
         dP[i][j]=dP[i][j-2]
      elif i>j and j>1:
         dP[i][j]=dP[i][j-2]+dP[i-j][j]
print(max(dP[-1][-1],dP[-1][-2]))
#dP[i][j]表示将i划分成最大奇数不大于j的划分数，dP[-1][-1]和dP[-1][-2]中一定有一个为0，另外一个才是正确答案

#回文字符串(特殊二维DP:dp[i][j]表示将片段s[i,j+1]改造成回文片段所需的最少操作次数)
s=list(input())
n=len(s)
dp=[[0]*n for _ in range(n)]
for l in range(2,n+1):
    for i in range(0,n):
        j=i+l-1
        if j<=n-1:
            if s[i]==s[j]:
                dp[i][j]=dp[i+1][j-1]
            else:
                dp[i][j]=min(dp[i][j-1],dp[i+1][j],dp[i+1][j-1])+1
print(dp[0][n-1])
#嵌套for循环的第一层是片段长度l(从2到len(s))，第二层是起始索引i(从0到len(s)-1)，
#相应地确定j=i+l-1
#初始dp矩阵全为0，保证i>=j的dp[i][j]均为0
#“增”和“删”是相同效果，对应dp[i][j-1]+1或dp[i+1][j]+1,“改”对应dp[i+1][j-1]+1

#欧拉筛：
max_num=int(input())
primes,check=[],[True for _ in range(2,max_num+1)]
for num in range(2,max_num+1):
    if check[num-2]:
        primes.append(num)
    for prime in primes:
        if prime*num>=max_num:
            break
        else:
            check[prime*num-2]=False
        if num%prime==0:
            break
#check列表的第一个元素是2
```



## <u>笔试CHEATING SHEET</u>

### 1.数据结构

| **数据结构：数据元素的集合**                                 |
| ------------------------------------------------------------ |
| **数据结构的三个基本要素:逻辑结构、存储结构以及基于结构定义的行为(运算)** |
| **数据元素：数据的基本单位，通常被视为一个逻辑上的实体，可以包含一个或多个数据项** |
| **数据项:构成数据元素的最小单位，是数据的基本部分。通常具有特定的类型或格式** |
| **数据的逻辑结构与数据元素本身的形式、内容、相对位置、个数无关** |
| **同一逻辑结构要求其包含的数据元素所包含的数据项的个数要相同，且对应数据项的类型要一致** |
| **数据结构设计影响算法效率，逻辑结构不起决定作用**           |
| **与数据元素本身的形式、内容、相对位置、个数无关的是数据的逻辑结构,存储结构/存储实现和运算实现与数据元素本身有关** |
| **逻辑结构：线性结构，树形结构，网格结构(图),集合结构(后三者统称非线性结构)** |
| **存储结构：顺序存储结构，链式存储结构，散列存储结构，索引存储结构** |



### 2.线性表

```python
#线性表/线性结构（逻辑结构的一种）：由顺序表（堆栈，队列，双端队列）和链表两种存储结构来实现
  #1.顺序表：连续存储结构==>访问元素的时间复杂度为O(1)，插⼊和删除元素的时间复杂度为O(n)
     #固定内存分配，内存空间连续
    #1.1堆栈：右进右出/左进左出==>应用：括号匹配问题，前/中/后序表达式求值及转换问题，
                                      #进制转换问题，用栈迭代实现DFS(深度优先搜索)
    #1.2队列：右进左出/左进右出==>应用：约瑟夫问题/BFS(广度优先搜索)
    #1.3双端队列：任意进任意出==>应用：回文数字
    #1.4循环队列(环形队列):
        #采用数组Q=[0..m-1]作为存储结构,其中变量rear表示循环队列中队尾元素的索引,也就是
        #队列中最晚入队的元素的索引，变量front表示队头元素的索引，也就是队列中现存最早入队的元素的索引
        #front=(1+rear+m-length)%m,变量length表示当前队列中的元素个数
        #在上述定义下，循环队列为空的条件为front==rear,(若rear定义不一样，条件也不同)
        #入队操作：添加元素时，首先将元素插入到队尾（rear），然后将rear指针向后移动一位，即rear=(rear+1)%m
        #如果rear移动到了数组的末尾，则将rear指针移到数组的开头
        #出队操作：删除元素时，首先将队头（front）的元素取出，然后将front指针向后移动一位，即front=(front+1)%m
        #如果front移动到了数组的末尾，则将front指针移到数组的开头
    #队头指针指向队列中最早入队的元素，队尾指针指向最晚入队的元素
  
  #2.链表：链式存储结构==>访问元素的时间复杂度为O(n)，插⼊和删除元素的时间复杂度为O(1)
  #(前提:已经知道要插入或删除的节点的指针或引用)，一般来说，从链表中删除或插入某个指定结点的时间复杂度是O(n)
     #动态内存分配，内存空间不需要连续
    #链表结构：由一系列节点组成，每个节点包括数据元素和指针两部分
    #链式存储结构所占存储空间分两部分，一部分存放结点值，一部分存放指示结点间关系的指针,所以存储密度一般小于1
    #所需空间与线性表长度不成正比
    #链表分类：根据指针的类型和连接方式，链表可以分为单向链表、双向链表和循环链表三种类型
    #若指定有n个元素的向量，则建立一个有序的单链表的时间复杂度为O(n*n)
    
  #串(字符串):特殊的线性表，是由零个或多个字符组成的有限序列，数据元素是一个字符   
  #有序表:特殊的线性表，其中的元素按照一定的规则或者顺序排列==>属于逻辑结构
  #有序表的二分查找(折半查找)过程中，若数组长度为偶数，取中间元素左边的元素为比较元素
  #Python中的list是动态数组，属于顺序表

#堆栈题目类型归纳：
  #用类写堆栈  
class Stack:
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items==[]
    def push(self,item):
        self.items.append(item)
    def peek(self):
        return self.items[len(self.items)-1]
    def pop(self):
        return self.items.pop()
    def size(self):
        return len(self.items)
    
  #十进制转换为其他进制
def base_converter(num_10,base):
    stack=[]
    digits='0123456789ABCDEF'
    while num_10>0:
        stack.append(num_10%base)
        num_10//=base
    num_base=''
    while stack:
        num_base+=digits[stack.pop()]
    return num_base

  #中序表达式转前/后序表达式
#Shunting Yard(调度场算法)是一种将中缀表达式转换为前/后缀表达式的算法
#主要思想是使用两个栈（运算符栈和输出栈）来处理表达式的符号
#按照运算符的优先级和结合性，将符号逐个处理并放置到正确的位置

#中缀表达式转后缀表达式的具体步骤:
#1.建立空运算符栈和输出栈
#2.从左到右遍历中缀表达式的每个符号
#    ①如果是操作数（数字/字母）:将其添加到输出栈
#    ②如果是左括号:将其推入运算符栈
#    ③如果是运算符:
#        如果该运算符的优先级大于运算符栈顶的运算符，或者运算符栈顶是左括号，将该运算符推入运算符栈
#        否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满足上述条件（或者运算符栈为空）
#        再将当前运算符推入运算符栈
#    ④如果是右括号:将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号，将左括号弹出。
#3.遍历完成后如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中

#Tip:将中缀表达式转换为前缀表达式的算法同理，只需将具体步骤中的"左"和"右"互换

  #用类写模拟器(以模拟器打印机为例)
import random

class Queue_:
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items==[]
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items.pop()
    def size(self):
        return len(self.items)

class Printer:
    def __init__(self,ppm):
        self.pagerate=ppm
        self.currentTask=None
        self.timeRemaining=0
    def tick(self):
        if self.currentTask!=None:
            self.timeRemaining=self.timeRemaining-1
            if self.timeRemaining==0:
                self.currentTask=None
    def busy(self):
        if self.currentTask!=None:
            return True
        else:
            return False
    def startNext(self,newtask):
        self.currentTask=newtask
        self.timeRemaining=newtask.getPages()*60/self.pagerate

class Task:
    def __init__(self,time):
        self.timestamp=time
        self.pages=random.randrange(1,21)
    def getStamp(self):
        return self.timestamp
    def getPages(self):
        return self.pages
    def waitTime(self,currenttime):
        return currenttime-self.timestamp

def simulation(numSeconds,pagesPerMinute):
    labprinter=Printer(pagesPerMinute)
    printQueue=Queue()
    waitingtimes=[]
    for currentSecond in range(numSeconds):
        if newPrintTask():
            task=Task(currentSecond)
            printQueue.enqueue(task)
        if (not labprinter.busy()) and (not printQueue.is_empty()):
            nexttask=printQueue.dequeue()
            waitingtimes.append(nexttask.waitTime(currentSecond))
            labprinter.startNext(nexttask)
        labprinter.tick()
    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining." % (averageWait, printQueue.size()))
    
def newPrintTask():   
    num=random.randrange(1,181)  #平均每180秒有一项新打印任务
    if num==180:    
        return True
    else:
        return False
    
for i in range(10):   #进行10次模拟
    simulation(3600,10)   #每次模拟的numSeconds==3600秒,ppm==10张/min

#双端队列题目类型归纳：
  #用类写双端队列
class Deque:
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items==[]
    def addRight(self,item):
        self.items.append(item)
    def addLeft(self,item):
        self.items.insert(0,item)
    def removeRight(self):
        return self.items.pop()
    def removeLeft(self):
        return self.items.pop(0)
    def size(self):
        return len(self.items)

#链表题目类型归纳：
  #单向链表实现
class Node:
    def __init__(self,value):
        self.value=value
        self.next=None

class LinkList:
    def __init__(self):
        self.head=None  #链表为空的条件
    def init_List(self,values):
        self.head=Node(values[0])
        current=self.head
        for value in values[1:]:
            node=Node(value)
            current.next=node
            current=current.next
    def insert(self,new_node):
        if self.head==None:
            self.head=new_node
        else:
            current=self.head
            while current.next:
                current=current.next
            current.next=new_node
    def lenth(self):
        current=self.head
        lenth=0
        while current:
            lenth+=1
            current=current.next
    def delete(self,node):
        if self.head==None:
            return None
        if self.head==node:
            self.head=self.head.next
        else:
            current=self.head
            while current.next:
                if current.next==node:
                    current.next=current.next.next
                    break
                current=current.next
    def reverse(self):   #反转单向链表法一
        current=self.head.next
        self.head.next=None 
        while current:
            pre=current
            current=current.next  			
            pre.next=self.head
            self.head=pre  
    def reverse_(self):  #反转单向链表法二
        pre=None
        current=self.head
        while current:
            next_node=current.next
            current.next=pre
            pre=current
            current=next_node
        self.head=pre
    def display(self):
        current=self.head
        while current:
            print(current.value,end=' ')
            current=current.next
        print()
        
  #双向链表实现
class Node_:
    def __init__(self,value):
        self.value=value
        self.next=None
        self.pre=None

class DoubleLinkList:
    def __init__(self):
        self.head=None
        self.tail=None
    def insert_front(self,node,new_node):
        if node==None:
            self.head=new_node
            self.tail=new_node
        else:
            new_node.next=node
            new_node.pre=node.pre
            if node.pre==None:
                self.head=new_node
            else:
                node.pre.next=new_node
            node.pre=new_node
    def display_forward(self):
        current=self.head
        while current:
            print(current.value,end=' ')
            current=current.next
        print()
        
  #循环链表实现
class Node__:
    def __init__(self,value):
        self.value=value
        self.next=None

class CircularLinkList:
    def __init__(self):
        self.head=None
    def insert_tail(self,new_node):
        if self.head==None:
            self.head=new_node
            new_node.next=self.head
        else:
            current=self.head
            while current.next!=self.head:
                current=current.next
            current.next=new_node
            new_node.next=self.head
    def insert_head(self,new_node):
        self.insert_tail(new_node)
        self.head=new_node
    def display(self):
        current=self.head
        while True:
            print(current.value,end=' ')
            current=current.next
            if current==self.head:
                break
        print()
    def delete(self,node): 
        if self.head==None:
            return None
        else:
            current=self.head
            while current.next!=node:
                current=current.next
                if current==self.head:
                    return False
            if current.next==self.head:
                self.head=self.head.next
            current.next=current.next.next
            return True
```



### 3.树

```python
#树是一种逻辑结构
#节点Node：节点是树的基础部分。每个节点具有名称，或“键值”。节点还可以保存额外数据项。
#边Edge：边是组成树的另一个基础部分。
  #每条边连接两个节点，表示节点之间具有关联，边具有出入方向；
  #每个节点（除根节点）有一条来自另一节点的入边；
  #每个节点可以有零条/一条/多条连到其它节点的出边。如果不能有 “多条边”，这里树结构就特殊化为线性表
#根节点Root: 树中唯一没有入边的节点。
#路径Path：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径所含节点个数为树的深度
#子节点Children：入边均来自于同一个节点的若干节点，称为这个节点的子节点。
#父节点Parent：一个节点是其所有出边连接节点的父节点。
#兄弟节点Sibling：具有同一父节点的节点之间为兄弟节点。
#子树Subtree：一个节点和其所有子孙节点，以及相关边的集合。
#叶节点Leaf Node：没有子节点的节点称为叶节点。
#层级Level：从根节点开始到达一个节点的路径，所包含的边的数量，称为这个节点的层级。
#高度/深度Height:树的最大层级
#节点的度:子节点数量，树的度：树中所有节点中最大的度数

#二叉树的层次遍历就是BFS,前序遍历就是DFS

#树的存储方式是链式存储，顺序存储不是树的存储形式,因为只有完全二叉树可以实现顺序存储

#森林（Forest）是指由多个树（Tree）组成的非连通的无环图
#森林的广度优先周游相当于对所有树一起进行层次遍历，深度优先周游相当于对每棵树单独DFS再相加

#把一棵树按照左儿子右兄弟的方式转置成二叉树，所得二叉树是唯一确定的
#把森林转置成二叉树，是先将每棵树转置成二叉树，再将所有二叉树合并起来
#设B是由一个森林变换得的二叉树，若森林中有n个非叶子节点，则B中右指针域为空的结点有n+1个

#二叉树的前序序列和后序序列相反，一定是所有节点均无左孩子或均无右孩子的二叉树，即只有1个叶子节点的二叉树

#所有叶结点在前序、中序和后序遍历结果中的相对次序不变

#若有一个叶子结点是二叉树的前序遍历结果的最后一个结点，它不一定是中序遍历结果的最后一个结点，
#因为根节点可能没有右子节点

#基于顺序表实现的逻辑结构不一定属于线性结构，还有可能是树

#树可以等价转化二叉树，树的先序遍历序列与其相应的二叉树相同

#二叉树的创建:
  #类创建法:
class Binarytree:
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
    def InsertLeft_(self,x_):
        if self.left==None:
            self.left=Binarytree(x_)
        else:
            t=Binarytree(x_)
            t.left=self.left
            self.left=t
    def GetLeftChild_(self):
        return self.left
    def GetRootVal_(self):
        return self.val
    def SetRootVal_(self,y):
        self.val=y
    def traversal(self,order):    #前、中、后序遍历二叉树
        if order=='preorder':
            print(self.key,end='')
        if self.left!=None:
            self.left.traversal(order)
        if order=='midorder':
            print(self.key,end='')
        if self.right!=None:
            self.right.traversal(order)
        if order=='postorder':
            print(self.key,end='')
            
  #列表创建法:
def BinaryTree(r):
    return [r,[],[]]
def GetLeftChild(root):
    return root[1]
def GetRootVal(root):
    return root[0]
def SetRootVal(root,new_val):
    root[0]=new_val
def InsertLeft(root,val):
    t=root.pop(1)
    if len(t)==0:
        root.insert(1,[val,[],[]])
    else:
        root.insert(1,[val,t,[]])
    return root

#解析树的创建:
   #将中序表达式转换为二叉解析树
#前置类
#class Stack(见线性表笔记)
#class Binarytree(上文已给出)
def parse_tree_mid(exp):
    lt=exp.split()
    stack=Stack()
    root=Binarytree('')
    stack.push(root)
    current=root
    for i in lt:
        if i=='(':
            stack.push(current)
            current.InsertLeft('')
            current=current.left
        elif i==')':
            current=stack.pop()
        elif i in '+-*/':
            current.SetRootVal(i)
            current.InsertRight('')
            stack.push(current)
            current=current.right
        else:
            current.SetRootVal(int(i))
            father=stack.pop()
            current=father
    return root
#Tips:该解析树是根据中序表达式建立的，对其进行前、后序遍历得到的结果不等于对应的前、后序表达式
  #中序遍历还原中序表达式
def print_mid_parse_tree(root):
    piece=''
    if root:
        piece+='('+print_mid_parse_tree(root.left)
        piece+=root.val
        piece+=print_mid_parse_tree(root.right)+')'
    return piece

  #计算二叉解析树
import operator
def evaluate(node):
    operators={'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv}
    l_node=node.left
    r_node=node.right
    if l_node and r_node:
        oper=operators[node.val]
        return oper(evaluate(l_node),evaluate(r_node))
    else:
        return node.val

  #将后序表达式转换为二叉解析树
def parse_tree_post(exp):
    stack=[]
    for i in exp:
        current=Node(i)
        if i in '+-*/':
            current.right=stack.pop()
            current.left=stack.pop()
        stack.append(current)
    return stack[0]
  #将前序表达式转换为二叉解析树:exp=list(reversed(exp)),其余代码同上
  #按层次遍历二叉解析树的结果，再将其反转就得到队列表达式
#根据二叉树前中序遍历求后序遍历
def post(pre,mid):
    if pre==[]:
        return ''
    Index=mid.index(pre[0])
    l_mid,r_mid=mid[:Index],mid[Index+1:]
    l_pre,r_pre=[],[]
    for i in pre[1:]:
        if mid.index(i)<Index:
            l_pre.append(i)
        else:
            r_pre.append(i)
    return post(l_pre,l_mid)+post(r_pre,r_mid)+pre[0]
#根据二叉树中后序遍历求前序遍历
def pre(mid,post):
    if mid==[]:
        return ''
    Index=mid.index(post[-1])
    l_mid,r_mid=mid[:Index],mid[Index+1:]
    l_post,r_post=[],[]
    for i in post[:-1]:
        if mid.index(i)<Index:
            l_post.append(i)
        else:
            r_post.append(i)
    return post[-1]+pre(l_mid,l_post)+pre(r_mid,r_post)
#Tips:根据二叉树前后序遍历无法得到确定的中序遍历

#二叉树建堆Binheap(笔试)==>完全二叉树
  #完全二叉树:所有的叶子节点都出现在最底层或者倒数第二层，最底层的叶子节点都集中在左边。
  #利用完全二叉树的性质来实现列表建树:
  #列表中索引为p的节点的左子节点处于索引2p，右子节点处于索引2p+1
class Binheap:
    def __init__(self):
        self.heaplist=[0]
        self.size=0
    def percUp(self,index):
        while index>1:
            if self.heaplist[index]<self.heaplist[index//2]:
                tmp=self.heaplist[index]
                self.heaplist[index]=self.heaplist[index//2]
                self.heaplist[index//2]=tmp
            index//=2
    def insert(self,x):
        self.size+=1
        self.heaplist.append(x)
        self.percUp(self.size)
    def percDown(self,index):
        while index*2<=self.size:
            mc=self.minChild(index)
            if self.heaplist[index]>self.heaplist[mc]:
                tmp=self.heaplist[index]
                self.heaplist[index]=self.heaplist[mc]
                self.heaplist[mc]=tmp
            index=mc
    def minChild(self,index):
        if (index*2+1)>self.size or self.heaplist[index*2]<self.heaplist[index*2+1]:
            return index*2
        return index*2+1          
    def minPop(self):
        Min=self.heaplist[1]
        self.heaplist[1]=self.heaplist[-1]
        self.size-=1
        self.heaplist.pop()
        self.percDown(1)
        return Min
    def Buildheap(self,lst):
        mid=len(lst)//2
        self.heaplist=[0]+lst
        self.size=len(lst)
        for index in range(mid,0,-1):  #因为叶子节点不可能下沉，所以从最后一个非叶子节点开始下沉
            self.percDown(index)

#二叉搜索树(二叉排序树/二叉查找树/BST树)具有二叉搜索性:
  #树高最小的二叉排序树搜索效率最好,时间性能为log2(n),和二分查找在数量级上一致，但一般的BST树达不到,甚至可能退化为O(n)
  #性质:小于父节点的节点都在左子树中，大于父节点的节点都在右子树中
  #适合对动态查找表进行高效率查找
  #节点的删除：
     #叶子节点：直接删除
     #只有一个子节点的节点：用其子节点替代，然后删除原先的子节点
     #有两个子节点的节点：找到该节点的后继节点（比待删除节点大的最小节点,即右子树中的最小节点）
     #或前驱节点（比待删除节点小的最大节点,即左子树中的最大节点）
     #来替代它，然后删除原先的后继节点或前驱节点
     #具体选择标准是让删除节点后的二叉树尽量平衡
#二叉搜索树按层次遍历:
def insert(root,val):
    if root==None:
        return Node(val)
    if val<root.val:
        root.left=insert(root.left,val)
    else:
        root.right=insert(root.right,val)
    return root

def level_order_traversal(root):
    queue=[root]
    traversal=[]
    while queue:
        current=queue.pop(0)  #也可用双端队列popleft
        traversal.append(current.value)
        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)
    return traversal

nums=[input()]
root=None
for num in nums:
    root=insert(root,num)
traversal=level_order_traversal(root)

#平衡二叉树(AVL树)==>能自动维持平衡的二叉搜索树
#对于任意节点,左子树和右子树的平衡因子之差不超过1,AVL树的高度保持在O(log n)的范围内,其中n是节点数量
#AVL树的插入、删除和搜索操作的时间复杂度均为O(log n)
#树型判断:LR型指的是平衡因子为2、-1的情形，其他情况同理
#为了保持AVL树的平衡,插入或删除节点时可能需要进行左旋、右旋、左右旋和右左旋
#具体地，LL型右旋，RR型左旋，LR型先左旋后右旋，RL型先右旋后左旋
  #左(右)倾二叉树的最坏情况是给定层数实现最少节点和给定节点数实现最多层数的情形
  #n层的AVL树至少有m个节点:递推公式m(n)=m(n-1)+m(n-2)+1
```



### 4.图论

```python
#字典的value如果是list，则为邻接表；dict嵌套dict则是字典树/前缀树/Trie
  #邻接表：使用字典表示图的邻接关系，将每个顶点的邻居顶点存储为列表
  #字典树（前缀树/Trie）：一种树形数据结构，用于高效地存储和检索字符串数据集中的键,
  #使用嵌套的字典来表示，每个字典代表一个节点，键表示路径上的字符，值表示子节点
  
#图Graph:由顶点Vertex和边Edge组成，每条边的两端都必须是图的两个顶点(可相同),分为有向图和无向图
#度Degree：度是指和某顶点相连的边的条数，对于有向图，顶点的出边条数称为出度，入边条数称为入度
#权值Weight：顶点和边所具有的量化的属性称为权值，分别称为点权和边权,根据问题的实际背景设定
#路径Path：由边连接的顶点组成的序列，有向图中起点和终点为同一顶点的路径称为环Cycle

#有向强连通图(广义的有向图成环，即2节点双向矢量路径也算作环):从图中任意一个顶点出发都能访问所有顶点，
#也就是说图中任意两个顶点之间都存在双向路径(注意:这不意味着所有的边都是成对反向的),
#有n个顶点的有向强连通图最少有n条边(即同向顺次连接，成环)
#强连通分量中任意2个顶点间均存在双向路径，且该分量是最大集合(1个顶点也可算1个强连通分量)
#连通分量是无向图中的极大连通子图，其中任意2顶点间存在路径

#生成树:原图的极小连通子图，包含所有顶点，且是树形结构（无环连通图）==>x个顶点，x-1条边
#图的BFS生成树的树高比DFS生成树的树高更小或相等
#由n个节点组成的有向无环图的最大有向边数为n*(n-1)/2
#用邻接矩阵法存储一个图时，不考虑压缩存储,存储空间大小只与图中结点个数有关，与图的边数无关

#图的实现方式
#1.邻接矩阵：适用于表示有很多条边的图
#2.邻接表：能够紧凑地表示稀疏图，可方便地找到与某顶点相连的其他所有顶点。
#3.关联矩阵：通常用于表示有向图
   
#用类实现图的邻接表
class Vertex:
    def __init__(self,key):
        self.key=key
        self.connect={}
    def addNeighbor(self,nbr,weight):
        self.connect[nbr]=weight
    def __str__(self):
        return str(self.key)+' connectedTo: '+str([x.key for x in self.connect])
    def getConnections(self):
        return self.connect.keys()
    def getId(self):
        return self.key
    def getWeight(self,nbr):
        return self.connect[nbr]

class Graph:
    def __init__(self):
        self.vertexs={}
        self.size=0
    def addVertex(self,key):
        self.size+=1
        newVertex=Vertex(key)
        self.vertexs[key]=newVertex
        return newVertex
    def getVertex(self,key):
        if key in self.vertexs.keys():
            return self.vertexs[key]
        else:
            return None
    def __contains__(self,key):  #布尔方法
        return key in self.vertexs
    def addEdge(self,key1,key2,weight):
        if key1 not in self.vertexs:
            self.addVertex(key1)
        if key2 not in self.vertexs:
            self.addVertex(key2)
        self.vertexs[key1].addNeighbor(self.vertexs[key2],weight)
        self.vertexs[key2].addNeighbor(self.vertexs[key1],weight)
        #若为有向边，则上述两段代码只取正向的一段
    def getVertexs(self):
        return self.vertexs.keys()
    def __iter__(self):  #定义迭代对象为Vertex(),应用:for vertex in Graph():
        return iter(self.vertexs.values())
    
#最小生成树(MST)==>x个顶点，(x-1)条边
#最小生成树是包含无向连通图中所有顶点的树,具有最小的总权值,同时保证n-1个边权值中的最大值最小
#如果一个连通无向图中所有边的权值均不同，则有唯一的最小生成树
#常见的建树算法有Prim算法和Kruskal算法：
#Prim算法：使用二叉堆+邻接表时间复杂度为O(ElogV)，使用简单数组+邻接矩阵的时间复杂度为O(V*V)
#选择一个起始顶点作为树的根节点
#找到与当前树中一个顶点相邻的所有树外顶点中边权值最小的一个，将其加入树中。
#更新生成树与树外顶点的距离。
#重复以上步骤，直到所有顶点都被加入，最终得到的树即为最小生成树
#Kruskal算法/并查集：时间复杂度为O(ElogV)
#将所有边按照权值从小到大排序
#从权值最小的边开始遍历，如果这条边连接的两个顶点不在同一个连通分量中，则将其加入最小生成树中，
#并合并这两个连通分量
#重复上述步骤，直到最小生成树中包含了图中的所有顶点
#Prim算法适用于稠密图，Kruskal算法适用于稀疏图

#判断有向图是否成环:拓扑排序
#拓扑排序是对有向无环图（DAG）排序的算法,包括Kahn算法和DFS(BFS也行，但效率更低)
#拓扑排序的结果是以顶点为元素的列表，任何有向路径都是从序列中索引小的顶点指向索引大的顶点
#常见的拓扑排序算法是Kahn算法:
#具体步骤:
#初始化：初始化一个用于存储结果序列的序列(可以有多个结果序列)和一个空集合（存储入度为0的顶点）
#计算入度：对于每个顶点，计算其入度，并将入度为0的顶点加入集合中
#循环处理：从集合中任意取出一个入度为0的顶点，并将其加入结果序列中，
#遍历以该顶点为起点的边，将这些边指向的顶点的入度减1。如果某个顶点的入度减为0，则将其加入集合中
#反复递归，直到集合为空
#检查环路：如果结果序列中包含了图中所有的顶点，则拓扑排序成功；否则说明图中存在环路，无法拓扑排序
n,m=map(int,input().split())
dic={i:[] for i in range(1,n+1)}  #存储以各节点为出发点可以指向的nbrs
lt=[0]*n   #存储各节点入度
for _ in range(m):
    x,y=map(int,input().split())
    dic[x].append(y)
    lt[y-1]+=1
ans,stack=[],[]
for i in range(n):
    if lt[i]==0:
        stack.append(i+1)
while stack:
    cur=stack.pop()  #此处可以任意取出一个入度为0的顶点，若要求字典序最小，则使用heapq
    ans.append(cur)
    for nbr in dic[cur]:
        lt[nbr-1]-=1
        if lt[nbr-1]==0:
            stack.append(nbr)
if len(ans)==n:
    print('拓扑排序成功')
else:
    print('存在环路，无法拓扑排序')
     
#Dijkstra算法：用于解决单源最短路径问题。基本思想是通过不断扩展离源节点最近的节点来逐步确定最短路径。
#具体步骤：(设节点数为n)
#初始化长度为n的数组，用于记录到所有节点的最短路径长，设到源节点的距离为0，其他节点的距离为无穷大。
#选择一个未访问的节点中距离最小的节点作为当前节点。
#遍历当前节点的邻居节点，如果通过当前节点到达邻居节点的路径比已知最短路径更短，则更新相应的最短路径。
#遍历完所有邻居节点后，标记当前节点为已访问。
#重复上述步骤，直到所有节点都被访问或者所有节点的最短路径都被确定。
#Dijkstra算法的时间复杂度为O(n^2)。
#当使用优先队列（如最小堆）来选择距离最小的节点时，可以将时间复杂度优化到O((n+E)logn)，
#其中E是图中的边数

#Bellman-Ford算法：用于解决单源最短路径问题，可以处理负权边。
#基本思想是通过松弛操作逐步更新节点的最短路径估计值，直到收敛到最终结果。
#具体步骤：(设节点数为n，边数为E)
#初始化长度为n的数组，用于记录到所有节点的最短路径长，设到源节点的距离为0，到其他节点的距离为无穷大。
#进行n-1次循环，每次循环对E条边(u,v)依次进行松弛操作:
#如果当前已知的从源节点到节点u的最短路径+边(u, v)的权重之和
#比当前已知的从源节点到节点v的最短路径更短，则更新相应最短路径。
#如果在n-1次循环后，仍然可以通过松弛操作更新最短路径，则说明存在负权回路，因此无法确定最短路径。
#Bellman-Ford算法的时间复杂度为O(n*E)。

#Floyd-Warshall算法，用于解决多源最短路径问题。
#Floyd-Warshall算法可以在有向图或无向图中找到任意两个顶点之间的最短路径。
#基本思想是用二维数组存储任意两个顶点之间的最短距离。
#初始时，这个数组包含图中各对单边连通的顶点之间的单边权重，对于不单边相连的顶点，权重为无穷大。
#迭代更新数组，逐步求得所有顶点之间的最短路径。
#具体步骤：
#初始化一个二维数组dist，用于存储任意两个顶点之间的最短距离。
#初始时，dist[i][j]表示顶点i到顶点j的直接边的权重，如果i和j不直接相连，则权重为无穷大。
#对于每个顶点k，考虑顶点k作为中间节点。
#遍历所有的顶点对(i, j),如果通过顶点k可以使得从顶点i到顶点j的路径变短，则更新dist[i][j]为更小的值。
#dist[i][j]=min(dist[i][j], dist[i][k]+dist[k][j])
#最终，dist数组中存储的就是所有顶点之间的最短路径。
#Floyd-Warshall算法的时间复杂度为O(n^3)。
#适用于稠密图的最短路径问题，并且可以处理负权边和负权回路。

#Kosaraju算法是一种用于在有向图中寻找强连通分量的算法，核心思想就是两次DFS。
#步骤如下:
#1. 第一次DFS：对图进行标准的DFS，在此过程中记录下顶点完成搜索的顺序。
#这一步的目的是为了找出每个顶点的完成时间。
#2. 反向图：接下来，我们对原图取反，将所有的边方向反转，得到反向图。
#3. 第二次DFS：在第二次DFS中，我们按照第一步中记录的顶点完成时间的逆序，
#对反向图进行DFS。找出反向图中的强连通分量。
def dfs1(graph, node, visited, stack):
    visited[node]=True
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs1(graph, nbr, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, ans):
    visited[node]=True
    ans.append(node)
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs2(graph, nbr, visited, ans)

def kosaraju(graph):
    stack=[]
    visited=[False]*n
    for node in range(n):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    tra_graph=[[] for _ in range(n)]
    for node in range(n):
        for nbr in graph[node]:
            tra_graph[nbr].append(node)
    visited=[False]*n
    answers=[]
    while stack:
        node=stack.pop()
        if not visited[node]:
            ans=[]
            dfs2(tra_graph, node, visited, ans)
            answers.append(ans)
    return answers
```



### 5.散列表

```python
#散列函数是一个多对一的映射
#散列是基于哈希表的逻辑结构
#(1)散列函数和散列地址：在记录存储位置p和其关键字key之间建立一个对应关系H，
#                      使p=H(key)，称对应关系H为散列函数，p为散列地址
#(2)散列表：一个有限连续的地址空间，用以存储按散列函数计算得到的散列地址，
#          通常散列表的存储空间是一个一维数组，散列地址是数组的下标
#(3)冲突和同义词：不同的关键字key可能对应同一散列地址,即p==H(key1)==H(key2)，这种现象称为冲突，
#                具有相同函数值的关键字称作同义词，key1与key2互称为同义词
#在散列表上进行查找的过程和创建散列表的过程一致
#由于冲突的发生，基于散列表的检索仍然需要进行关键码对比，关键码的比较次数
#不仅仅取决于构造的散列函数与处理冲突的方法两个因素，还和关键码本身有关

#散列查找法(即哈希查找)主要研究以下两方面的问题:
#(1)构造散列函数
#1.数字分析法
#适用情况：事先必须明确知道所有的关键字每一位上各种数字的分布情况
#从关键字中提取数字分布比较均匀的若干位作为散列地址
#2.平方取中法
#一个数平方后的中间几位数和数的每一位都相关，如果取关键字平方后的中间几位或其组合
#作为散列地址，则使随机分布的关键字得到的散列地址也是随机的，具体所取的位数由表长决定
#3.折叠法
#适用情况：散列地址的位数较少，而关键字的位数较多，且难于直接从关键字中找到取值较分散的几位
#将关键字按若干位数一段分割成几部分，然后取这几部分的叠加和（舍去进位）作为散列地址
#折叠法分为移位叠加和边界叠加两种
#4.除留余数法（最常用）
#假设散列表表长为m，选择一个不大于m的数p（一般选小于表长的最大质数），令H(key)=key%p
#可以对关键字直接取模，也可在折叠、平方取中等运算之后取模

#(2)处理冲突
#1.开放地址法
#当某关键字key的初始散列地址H0==H(key)发生冲突时，以H0为基础，采取合适方法计算得到
#另一地址H1，如果H1仍然发生冲突，以H1为基础再求下一个地址H2，依次类推，直至Hk不发生冲突为止，
#则Hk为最终确定的散列地址
#Hi=[H(key)+di]%m,i=1,2,…,k(k≤m-l)
#寻找空位的过程称为探测
  #线性探测法:di=i
  #二次探测法:di=1^2,-1^2,2^2,-2^2,3^2,…,k^2,-k^2
  #伪随机探测法:di=伪随机数序列
  #双重散列探测法：反复使用*再散列函数*作为步长，直到找到一个空位
#2.链地址法
#把具有相同散列地址的记录放在同一单链表中，称为同义词链表
#有m个不同散列地址就有m个同义词链表，用数组HT[0…m-1]存放各链表的头指针，
#散列地址为i的记录都以结点方式插入到以HT[i]为头结点的单链表中
#3.再散列（Rehashing）：当散列表的装载因子（load factor,即占用率）超过一定阈值时，进行扩容操作，
#调整散列函数和同义词链表的数量，以减少冲突的概率
#4.建立公共溢出区（Public Overflow Area）：将冲突的元素存储在一个公共的溢出区域，
#而不是在同义词链表中，进行查找时遍历溢出区域

#散列表代码实现:
class HashTable:
    def __init__(self):
        self.size=11  #size最好是素数
        self.slots=[None]*self.size
        self.data=[None]*self.size

    def put(self,key,data):
        hashvalue=self.hashfunction(key,len(self.slots))
        if self.slots[hashvalue]==None:
            self.slots[hashvalue]=key
            self.data[hashvalue]=data
        else:
            if self.slots[hashvalue]==key:
                self.data[hashvalue]=data #替换原有data
            else:
                nextslot=self.rehash(hashvalue,len(self.slots))
                while self.slots[nextslot]!=None and self.slots[nextslot]!=key:
                    nextslot=self.rehash(nextslot,len(self.slots))
                if self.slots[nextslot]==None:
                    self.slots[nextslot]=key
                    self.data[nextslot]=data
                else:
                    self.data[nextslot]=data #替换原有data

    def hashfunction(self,key,size):
        return key%size

    def rehash(self,oldhash,size):
        return (oldhash+1)%size

    def get(self,key):
        startslot=self.hashfunction(key,len(self.slots))
        data=None
        stop=False
        found=False
        position=startslot
        while self.slots[position]!=None and not found and not stop:
                if self.slots[position]==key:
                    found=True
                    data=self.data[position]
                else:
                    position=self.rehash(position,len(self.slots))
                    if position==startslot:
                        stop=True
        return data

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)
        
#MD5加密
import hashlib

def calculate_md5(text):
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode('utf-8'))
    return md5_hash.hexdigest()

n=int(input())
for _ in range(n):
    text1=input()
    text2=input()
    md5_text1=calculate_md5(text1)
    md5_text2=calculate_md5(text2)
    if md5_text1==md5_text2:
        print("Yes")
    else:
        print("No")
```



### 6.排列

```python
#直接插入排序、冒泡排序、希尔排序都是在数据正序的情况下比逆序的情况下要快
#归并排序是先对最小分区排序，逐次归并到整体序列；快速排序是先对整体交换排序，逐次分区到最小
#第几次分区等于第几次调用函数，一轮排序等于对相同级别分区都进行排序
#第一轮排序过后，只有插入排序无法得到任一元素的最终排序位置
#归并排序是外部排序算法，适合主存储器的可用空间有限的情况。
#分治和动态规划的区别：分治法解决的问题不拥有重叠子问题，而动态规划解决的问题拥有重叠子问题和最优子结构
#分治算法将原问题分解为几个规模较小但类似于原问题的子问题，但不要求算法实现一定写成递归，还可以直接求解
#递归是函数的反复调用，迭代是循环结构的反复进行

#插入排序原理：
#初始状态：假设第一个元素已经是一个有序序列，从第二个元素开始，将其作为待插入元素。
#插入操作：对于每个待插入元素，将其与已排序序列中的元素从后往前依次比较，找到合适的插入位置，
#并将其插入到合适位置后。
#重复插入操作，直到所有元素都被插入到合适的位置
arr=list(map(int,input().split()))
for i in range(1,len(arr)):
    key=arr[i]
    j=i-1
    while j>=0 and arr[j]>key:
        arr[j+1]=arr[j]
        j-=1
    arr[j+1]=key

#希尔排序（Shell Sort）是插入排序的改进版本，也称为缩小增量排序
#通过将原始数组分成多个子序列进行插排，随着排序的进行，逐渐减小子序列长度，最终完成整个数组的排序
#希尔排序的核心思想是先通过较大的步长进行排序，然后逐渐缩小步长直至步长为1，
#最后进行一次完整的插入排序

#选择排序(SelectionSort)==>堆排序本质上是选择排序
#选择排序原理：
#初始状态：将整个序列看作是未排序部分和已排序部分两部分，初始时已排序部分为空。
#选择最小元素：从未排序部分中选择最小的元素，并与未排序部分的第一个元素交换位置，将其放置到已排序序列的末尾。
#重复步骤，直到所有元素都被放置到已排序序列的合适位置。
arr=list(map(int,input().split()))
for i in range(len(arr)):
    min_index=i
    for j in range(i+1,len(arr)):
        if arr[min_index]>arr[j]:
            min_index=j
    arr[i],arr[min_index]=arr[min_index],arr[i]
    
#交换排序包括冒泡排序和快速排序:
#冒泡排序(BubbleSort)
arr=list(map(int,input().split()))
lenth=len(arr)
for i in range(lenth):
    check=True
    for j in range(lenth-i-1):
        if arr[j]>arr[j+1]:
            arr[j],arr[j+1]=arr[j+1],arr[j]
            check=False
    if check:
        break
    
#快速排序(QuickSort)
#快速排序在被排序的数据完全无序的情况下优势最明显
arr=list(map(int,input().split()))
#单指针快排:
def QuickSort_1(arr):
    if len(arr)<=1:
        return arr
    else:
        pivot=arr[0]
        l_arr=[i for i in arr[1:] if i<pivot]
        r_arr=[j for j in arr[1:] if j>=pivot]
    return QuickSort_1(l_arr)+[pivot]+QuickSort_1(r_arr)

#双指针快排(Lomuto分区):
    #以最右端元素为pivot
def Partition(arr,low,high):
    pivot=arr[high]
    i=low-1
    for j in range(low,high):
        if arr[j]<=pivot:
            i+=1
            arr[i],arr[j]=arr[j],arr[i]
    arr[i+1],arr[high]=arr[high],arr[i+1]
    return i+1

    #以最左端元素为pivot
def partition(arr,low,high):
    pivot=arr[low]  
    i=low+1
    j=high
    while True:
        while i<=j and arr[i]<=pivot:
            i+=1
        while i<=j and arr[j]>=pivot:
            j-=1
        if i<=j:
            arr[i],arr[j]=arr[j],arr[i]
        else:
            break
    arr[low],arr[j]=arr[j],arr[low]
    return j

def QuickSort_2(arr,low,high):
    if low<high:
        pi=Partition(arr,low,high)
        QuickSort_2(arr,low,pi-1)
        QuickSort_2(arr,pi+1,high)
#初始low=0,high=len(arr)-1
#3中位数法:pivot选取arr[low],arr[high],add[mid](mid=(low+high+1)//2)中的中位数

#归并排序(MergeSort)
arr=list(map(int,input().split()))
def MergeSort(arr):
    if len(arr)==1:
        return arr
    mid=int(len(arr)/2)
    l_arr,r_arr=MergeSort(arr[:mid]),MergeSort(arr[mid:])
    i,j=0,0
    #ans=0
    arr_sort=[]
    while i<len(l_arr) and j<len(r_arr):
        if l_arr[i]<r_arr[j]:
            #ans+=(len(arr_l)-i)   #逐个比较求逆序数
            arr_sort.append(l_arr[i])
            i+=1
        else:
            arr_sort.append(r_arr[j])
            j+=1
    arr_sort+=l_arr[i:]+r_arr[j:]
    return arr_sort

#拓扑排序(见图论)

#求逆序数
  #法一:bisect二分插入
from bisect import bisect_left,insort_left
while True:
    n=int(input())
    if n==0:
        break
    a=[]
    rev=0
    for _ in range(n):
        num=int(input())
        rev+=bisect_left(a,num)
        insort_left(a,num)
    ans=n*(n-1)//2-rev
    print(ans)
  #法二:归并排序(见上文代码，ans即为答案)
```

#### 排序方法总结表

![排序总结.png](https://s2.loli.net/2024/06/20/eqw7NOyI89AvtEz.png)

