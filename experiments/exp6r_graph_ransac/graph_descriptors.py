from __future__ import annotations
from collections import Counter,deque
import numpy as np
from .canonical import canonical_edge

def _distances(n,adj,start):
    d=[999]*n; d[start]=0; q=deque([start])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if d[v]==999: d[v]=d[u]+1;q.append(v)
    return d

def describe_graph(graph):
    nodes=np.asarray(graph["nodes"],float); edges=[tuple(map(int,e)) for e in graph["edges"]]; n=len(nodes)
    adj=[set() for _ in range(n)]
    for a,b in edges: adj[a].add(b);adj[b].add(a)
    degrees=[len(x) for x in adj]; centre=np.median(nodes,axis=0); scale=max(np.median(np.linalg.norm(nodes-centre,axis=1)),1e-9)
    node_sig=[]
    for i in range(n):
        neighbor_degrees=tuple(sorted(degrees[j] for j in adj[i])); hist=tuple(sorted(Counter(neighbor_degrees).items()))
        paths=_distances(n,adj,i); path_hist=tuple(sorted(Counter(x for x in paths if x<999).items()))
        radial=round(float(np.linalg.norm(nodes[i]-centre)/scale),5)
        distance_profile=tuple(sorted(round(float(x),5) for x in np.linalg.norm(nodes-nodes[i],axis=1)/scale))
        node_sig.append((degrees[i],hist,path_hist,radial,distance_profile))
    lengths=np.asarray([np.linalg.norm(nodes[a]-nodes[b])/scale for a,b in edges],float)
    order=np.argsort(np.argsort(lengths,kind="stable"),kind="stable") if len(lengths) else []
    result=[]
    for k,(a,b) in enumerate(edges):
        u,v=canonical_edge(a,b,node_sig); du,dv=degrees[u],degrees[v]
        junction=[]
        for x in (u,v):
            ds=_distances(n,adj,x); junction.append(min((ds[j] for j,d in enumerate(degrees) if d>=3),default=999))
        result.append({"endpoints":(u,v),"endpoint_signatures":tuple(sorted((node_sig[u],node_sig[v]))),
          "degree_multiset":tuple(sorted((du,dv))),"normalized_length":float(lengths[k]),
          "length_rank":float(order[k]/max(len(edges)-1,1)),"neighbor_degree_histograms":tuple(sorted((node_sig[u][1],node_sig[v][1]))),
          "shortest_path_context":tuple(sorted((node_sig[u][2],node_sig[v][2]))),
          "type":"terminal" if min(du,dv)==1 else ("branching" if max(du,dv)>=3 else "internal"),
          "junction_distances":tuple(sorted(junction)),"canonical_key":(tuple(sorted((node_sig[u],node_sig[v]))),round(float(lengths[k]),5))})
    return {"node_signatures":node_sig,"edges":sorted(result,key=lambda x:x["canonical_key"]),"scale":scale}
