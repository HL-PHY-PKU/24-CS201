#图论

#有向强连通图(广义的有向图成环，即2节点双向矢量路径也算作环):从图中任意一个顶点出发都能访问所有顶点，
#也就是说图中任意两个顶点之间都存在双向矢量路径(注意:这不意味着所有的边都是成对反向的),
#有n个顶点的有向强连通图最少有n条边(即同向顺次连接，成环)
#由n个节点组成的有向无环图的最大有向边数为n*(n-1)/2
#图的实现方式
#1.邻接矩阵：适用于表示有很多条边的图
#2.邻接表：能够紧凑地表示稀疏图，可方便地找到与某顶点相连的其他所有顶点。
#3.关联矩阵：通常用于表示有向图

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
#常见的建树算法有Prim算法和Kruskal算法：
#Prim算法：时间复杂度为O(mlogn)，m为边的数量，n为顶点数量
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
#Kruskal算法/并查集：时间复杂度为O(ElogE),E为边的数量
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
#Dijkstra算法的时间复杂度为O(n^2)。
#当使用优先队列（如最小堆）来选择距离最小的节点时，可以将时间复杂度优化到O((n+E)logn)，
#其中E是图中的边数
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

#线性表

#线性表/线性结构（逻辑结构的一种）：由顺序表（堆栈，队列，双端队列）和链表两种存储结构来实现
  #1.顺序表：连续存储结构
    #1.1堆栈：右进右出/左进左出==>应用：括号匹配问题，前/中/后序表达式求值及转换问题，
                                      #进制转换问题，用栈迭代实现DFS(深度优先搜索)
    #1.2队列：右进左出/左进右出==>应用：约瑟夫问题/BFS(广度优先搜索)
    #1.3双端队列：任意进任意出==>应用：回文数字
    #1.4循环队列(环形队列):
        #采用数组Q=[0..m-1]作为存储结构,其中变量rear表示循环队列中队尾元素的索引,也就是
        #队列中最晚入队的元素的索引，变量front表示队头元素的索引，也就是队列中现存最早入队的元素的索引
        #front=(1+rear+m-length)%m,变量length表示当前队列中的元素个数
        #在上述定义下，front初始为0，rear初始为-1
        #入队操作：添加元素时，首先将元素插入到队尾（rear），然后将rear指针向后移动一位，即rear=(rear+1)%m
        #如果rear移动到了数组的末尾，则将rear指针移到数组的开头
        #出队操作：删除元素时，首先将队头（front）的元素取出，然后将front指针向后移动一位，即front=(front+1)%m
        #如果front移动到了数组的末尾，则将front指针移到数组的开头
    #队头指针指向队列中最早入队的元素，队尾指针指向最晚入队的元素
  
  #2.链表：链式存储结构
    #链表结构：由一系列节点组成，每个节点包括数据元素和指针两部分
    #链表分类：根据指针的类型和连接方式，链表可以分为单向链表、双向链表和循环链表三种类型

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

  #单向链表实现
class Node:
    def __init__(self,value):
        self.value=value
        self.next=None

class LinkList:
    def __init__(self):
        self.head=None
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

#二叉树

#路径Path：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，
#层级Level：从根节点开始到达一个节点的路径，所包含的边的数量，称为这个节点的层级。
#高度/深度Height:树的最大层级
#把一棵树按照左儿子右兄弟的方式转置成二叉树，所得二叉树是唯一确定的

  #前、中、后序遍历二叉树
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
#4.1根据二叉树前中序遍历求后序遍历
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
#4.2根据二叉树中后序遍历求前序遍历
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

#6.二叉树建堆Binheap(笔试)==>完全二叉树
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
        for index in range(mid,0,-1):
            self.percDown(index)

#7.二叉搜索树(二叉排序树/二叉查找树/BST树)具有二叉搜索性:
  #树高最小的二叉排序树搜索效率最好,时间性能为log2(n),和二分查找在数量级上一致，但一般的BST树达不到
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

#8.平衡二叉树(AVL树)==>能自动维持平衡的二叉搜索树
#对于任意节点,左子树和右子树的平衡因子之差不超过1,AVL树的高度保持在O(log n)的范围内,其中n是节点数量
#AVL树的插入、删除和搜索操作的时间复杂度均为O(log n)
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

#9.并查集(Disjoint Set)
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

#给出伪满二叉树(用#等符号代替叶子节点的空子节点)的前序遍历，要求建树:
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

#排序

#插入排序原理：
#初始状态：假设第一个元素已经是一个有序序列，从第二个元素开始，将其作为待插入元素。
#插入操作：对于每个待插入元素，将其与已排序序列中的元素从后往前依次比较，找到合适的插入位置，
#并将其插入到合适位置后。
#重复插入操作，直到所有元素都被插入到合适的位置

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
#快速排序在被排序的数据完全无序的情况下最易发挥其长处
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

#双指针快排(霍尔法):
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

#希尔排序（Shell Sort）是插入排序的改进版本，也称为缩小增量排序
#通过将原始数组分成多个子序列进行插排，随着排序的进行，逐渐减小子序列长度，最终完成整个数组的排序
#希尔排序的核心思想是先通过较大的步长进行排序，然后逐渐缩小步长直至步长为1，
#最后进行一次完整的插入排序

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