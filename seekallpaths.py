# -*- coding: utf-8 -*-
"""
seekallpaths.py - ΔΙΟΡΘΩΜΕΝΗ ΕΚΔΟΣΗ

Η ΜΟΝΗ αλλαγη ειναι ο ΤΡΟΠΟΣ ΥΠΟΛΟΓΙΣΜΟΥ ΤΟΥ ΧΡΟΝΟΥ μιας ακολουθιας κομβων,
σε 4 σημεια (0, 1, 2 και 3 μετεπιβιβασεις).

ΔΕΝ αλλαζει: η λογικη ευρεσης μονοπατιων, οι ποινες μετεπιβιβασης (+5/+10/+15),
η δομη των επιστρεφομενων tuples, οι υπογραφες, τιποτα αλλο.

Το σφαλμα:  time += TT[node][l[l.index(node)+1]]
Το l.index() επιστρεφει την ΠΡΩΤΗ εμφανιση. Σε [39,40,41,40,39] εδινε 12.545
αντι για 18.083 λεπτα (-31%). Στις αποθηκευμενες λυσεις υπηρχαν 21 τετοιες
διαδρομες, και το αρχικο routeset περιεχει ηδη 3.

Η διορθωση: time = sum(TT[l[k]][l[k+1]] for k in range(len(l)-1))
"""
import heapq


def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False
    
def shortestPath(graph, start, end):
    queue = [(0, start, [])]
    seen = set()
    while True:
        (cost, v, path) = heapq.heappop(queue)
        if v not in seen:
            path = path + [v]
            seen.add(v)
            if v == end:
                return cost, path
            for (next, c) in graph[v].items():
                if next not in seen:
                  heapq.heappush(queue, (cost + c, next, path))
                  
def seekpathfr(itest, ori,d,routeset,TT,trstops,sp,doit):
    paths=[]
#paths holds time,sequence and transfers
    oinroutes=[]
    dinroutes=[]
    #IMPORTANT

    for  idx,route in enumerate(routeset):
               
               
                #print route
                #check if SP already in some route
                #reverses route
               revr=route[::-1]
               if ori in route:
                 oinroutes.append(idx)
                 if d in route: 
                    if (contains(sp[1],revr)!=False) or (contains(sp[1],route)!=False):
                        s=(sp[0],sp[1],0,[idx, route.index(ori),route.index(d)])
                        #list(sp.extend(0))
                        #paths.append(s)
                        heapq.heappush(paths,s)
                        #break----NOT BREAKING TO FIND ALTERNATE PATHS
                        #continue
                    #break
                        #print paths
                    else:
                        #check if nodes directly in route (order may not be sp)
                        time=0
                        path2=[]
                        start=route.index(ori)
                        end=route.index(d)
                        #SP[1][SP[1].index(node2)+1]
                        if start<end:
                            l=route[start:end+1]
                        else:
                            l=route[end:start+1]
                           # l=l[::-1]
                            #print l   
                            l.reverse()
                        # [ΔΙΟΡΘΩΣΗ 1/4] Ο παλιος βροχος χρησιμοποιουσε l.index(node),
                        # που επιστρεφει ΠΑΝΤΑ την πρωτη εμφανιση του κομβου. Σε διαδρομες
                        # με επαναλαμβανομενο κομβο (π.χ. [4,5,6,5,8]) εδινε λαθος διαδοχο
                        # και λαθος χρονο. Επισης το "node != l[-1]" συγκρινε ΤΙΜΗ αντι για
                        # θεση, οποτε σταματουσε νωρις αν ο τελευταιος κομβος εμφανιζοταν
                        # και νωριτερα. Τελος, αν len(l)==1 το score1 δεν οριζοταν ποτε και
                        # το heappush επετασσε NameError.
                        # Ο νεος υπολογισμος αθροιζει τις ΔΙΑΔΟΧΙΚΕΣ ΘΕΣΕΙΣ - ταυτοσημος
                        # με την προθεση του αρχικου κωδικα, χωρις τα τρια σφαλματα.
                        if len(l) >= 2:
                            path2 = list(l)
                            time = sum(TT[l[k]][l[k+1]] for k in range(len(l)-1))
                            score1 = (time, path2, 0, [idx, start, end])
                            heapq.heappush(paths, score1)
                        #paths.append(score)
             #POSSIBLE BREAK 
                 #if len(paths)!=0:break #why find transf if direct path? ALSO IF i have shortest path stop looking
                 
                 for  idx2,route2 in enumerate(routeset):
                        if d in route2  and idx2 not in dinroutes:  
                            dinroutes.append(idx2)
                            #print route2,route
                        #print routeset
               #print oinroutes,dinroutes
               #print dinroutes   
    l1=[]
    l2=[]
    l=[]  
            #sc=[]  
             #NOW FOR 1 1 trans
    for i in oinroutes:
                    cur=routeset[i]
                    start=cur.index(ori) #holds position of origin in first route
                    for j in dinroutes:
                      if i!=j and cur!=routeset[j]:
                        cur2=routeset[j]
                        end2=cur2.index(d) #Holds position of destination in last route
                        #possible trans points for routes index(i) in o and indexj in d i*j pairs
                        points=trstops[i][j]
                        for point in points:
                           # print point,cur
                            if point==ori or point==d: continue # ALREADY SEEN IN ZERO TRANSF
                            end=cur.index(point)  #position of each transfer point in first route
                            #[routeset[i][routeset.index(ori)]]
                           # start=routeset[i][routeset[routeset.index(ori)]]
                            #end=routeset[i][routeset.index(point)]
                            #try 8 14 5
                            if start<end:
                                l1=cur[start:end+1]
                            else:#if trans index is before origin index
                                l1=cur[end:start+1]
                                #then reverse
                                l1=l1[::-1]
                            #print l1, cur, start, end, point, ori
                            
                            start2=cur2.index(point) #position of eqach trasnf point in last route
                            
                            
                            if start2<end2:
                                l2=cur2[start2:end2+1]
                            else:#if destination index is before trans index
                                l2=cur2[end2:start2+1]
                                l2=l2[::-1]
                            #print l2, cur2, start2, end2, point, d
            #print routeset 
                            #print list(set(l2).intersection(l1)) IF THERE IS ONLY ONE COMMON POINT BETWEEN THE TWO STRINGS
                            if len(set(l2).intersection(l1))==1:
                                #print l2,l1
                                if l1[-1]==l2[0]:
                                    
                                    common=list(set(l2).intersection(l1))
                                    l1.remove(l1[-1])
                                    l=l1+l2
                               #print l1,l2,l
                                elif l1[0]==l2[-1]:
                                    common=list(set(l2).intersection(l1))
                                    l2.remove(l2[-1])
                                    l=l2+l1
                               #exit
                                # [ΔΙΟΡΘΩΣΗ 2/4] ιδιο σφαλμα l.index() - 1 μετεπιβιβαση.
                                # Η ποινη +5 λεπτων ανα μετεπιβιβαση ΔΕΝ αλλαζει.
                                path3 = list(l)
                                time = sum(TT[l[k]][l[k+1]] for k in range(len(l)-1)) if len(l) >= 2 else 0
                                sc=(time+5,path3,1,[i,start,end,j,start2,end2])
                               # print sc[0]
                                #if len(paths)!=0: print paths[0][0]
                                #do not need to add higher values
                                #if len(paths)!=0 and sc[0]>paths[0][0]:break
                                #print sc
                                if sc not in paths: heapq.heappush(paths,sc)                    
                                #paths.append(sc)
                                #print paths
                                
            #POSSIBLE BREAK BEFORE 2 TRANS
                        # !!!if paths:
                        #      if paths[0][2]==0:continue #THERE IS NO REASON TO LOOK FOR 2 TRNSF for this i, j routes
             #if len(paths)!=0:break #if found one transfer no need to find two ? NOT ALWAYS EG OD 2-8 seed 4
                    #now check all intersections of routes FOR 2 TRANSF
                    # inside if i!=j so that cu kai cur 2 are defined along with start end2
                        #remove or route and d route
                        seen=set([i,j])
                        #print seen
                        e=[u for u in range(len(routeset))]
                        #print e
                        posroutes=list(set(e).difference(seen))
                        numberofint=len(posroutes) #how many possible interm routes
                        #print numberofint, list(set(e).difference(seen))
                        
                        #k in index of routes not or or dest
                        for k in posroutes:
                        #for k in range(len(routeset)):
                              #print k
                          #if k!=i and k!=j: #k is the intermediate route
                          #if k==i or k==j: continue
                          if (routeset[k]!=routeset[j]) and (routeset[k]!=routeset[i]) and (routeset[i]!=routeset[j]):
                              #print i
                              group1=trstops[i][k]
                              group2=trstops[k][j]
                              interm=routeset[k] # FIRST POSS INTERM ROUTE
                              #IF NOT ALL TRNS POINTS COMMON IN 2 SETS, OTHERWISE DO ONE TRANSF
                              if len(set(group1).intersection(group2))>0:
                                  #print group1,group2, k,i,j
                                  for p1 in group1:
                                      #first find position of each point in origin route cur
                                      endinor=cur.index(p1)
                                      if start<endinor:
                                          l11=cur[start:endinor+1]
                                      else:#if trans index is before origin index
                                          l11=cur[endinor:start+1]
                                          #then reverse
                                          l11=l11[::-1]
                                      #now find position of this point in interm route
                                      startint=interm.index(p1)
                                     #now find possible points for jumping to destination route cur2
                                      for p2 in group2:
                                         #if it is in both transf sets then it can be done in one transf ALREADY SEEN BEFORE
                                         if p2 not in group1:
                                             endint=interm.index(p2)
                                             if startint<endint:
                                                 l21=interm[startint:endint+1]
                                             else:#if trans index is before origin index
                                                 l21=interm[endint:startint+1]
                                                 #then reverse
                                                 l21=l21[::-1]
                                                 
                                            #NOW find position of 2nd trns point in dest route cur2
                                             startindr=cur2.index(p2)
                                             if startindr<end2:
                                                 l31=cur2[startindr:end2+1]
                                             else:#if trans index is before origin index
                                                 l31=cur2[end2:startindr+1]
                                                 #then reverse
                                                 l31=l31[::-1]
                                                 
                                             #NOW check if only one common point and no cycles between sub strings    
                                             if len(set(l11).intersection(l21))==1 and len(set(l21).intersection(l31))==1 and len(set(l11).intersection(l31))==0:
                                #merge substrings
                                                if l11[-1]==l21[0]:
                                                    common1=list(set(l21).intersection(l11))
                                                    l11.remove(l11[-1])
                                                    ln=l11+l21
                                                elif l11[0]==l21[-1]:
                                                    common1=list(set(l21).intersection(l11))
                                                    l21.remove(l11[-1])
                                                    ln=l21+l11
                                                if ln[-1]==l31[0]:
                                                                      
                                                    common2=list(set(ln).intersection(l31))
                                                    l31.remove(l31[0])
                                                    ln=ln+l31
                                                elif ln[0]==l31[-1]:
                                                    common2=list(set(ln).intersection(l31))
                                                    l31.remove(l31[-1])
                                                    ln=l31+ln
                                                #print l11,l21,l31,common1,common2,ln
                                               #exit
                                                # [ΔΙΟΡΘΩΣΗ 3/4] ιδιο σφαλμα - 2 μετεπιβιβασεις.
                                                # Η ποινη +10 λεπτων ΔΕΝ αλλαζει.
                                                path4 = list(ln)
                                                time = sum(TT[ln[k]][ln[k+1]] for k in range(len(ln)-1)) if len(ln) >= 2 else 0
                                                sc1=(time+10,path4,2,[i,start,endinor,k,startint,endint,j,startindr,end2])
                                                #print sc[0]
                                                #if len(paths)!=0: print paths[0][0]
                                                #do not need to add higher values
                                                #print paths
                                                #if len(paths)!=0 and sc1[0]>paths[0][0]:break
                                                if sc1 not in paths: heapq.heappush(paths,sc1)
                                                #paths.append(sc1)
                                                #print heapq.heappop(paths) 
                                                #print paths[0]
                                                #ODPATHS[ori][d]=paths[0]  
                              #print   ODPATHS[ori][d]  
                            ####### 3 or MORE TRANSF
               ###try to find for 8-2 path 8-14 14-6-9-12  12-10-11-3  3-5-2  with 3 trns
            #if len(paths)==0 :   #look for more than 2 transf
            #add lists to concatenate
                        #print paths
                                                
                       # print paths
                        #raw_input("Press Enter to continue...")
                        #if paths[0][2]==0:continue #THERE IS NO REASON TO LOOK FOR 3 TRNSF
#doit=1
    #if paths[0][2]==0:doit=0                                            
    if len(paths)==0 or doit==1 :
        for i in oinroutes:
            for j in dinroutes:
                if i!=j:# NOW SET and routeset[i]!=routeset[j]:
                           seen=(i,j)
                        #print seen
                           e=[u for u in range(len(routeset))]
                        #print e
                           #posroutes=routeset.remove(seen)
                           posroutes=list(set(e).difference(seen))
                           #print seen,posroutes
                           count=1
                           comb=[]
                           #print paths
                           #raw_input("Press Enter to continue...")
                           if not posroutes:continue# IF ITS EMPTY
                           for h in range(len(posroutes)):
                                                #print h
                                                comb.append(posroutes[h])
                                                c1=[]
                                                if posroutes[h]!=posroutes[-1]:
                                                    for t in range(h,len(posroutes)):
                                                        c1.append(posroutes[t])
                                                        #print c1
                                                    comb.append(c1)
                                                    comb.append(c1[::-1])
                                                #print comb#[1][0]
                                                #create lists 
                                                combod=[]
                                                #help(comb)
                           for k in range(len(comb)):
                                                  #c=[]if isinstance(e, list):
                                                  if isinstance(comb[k],list):
                                                      c=comb[k]
                                                      #for t in range(len(comb[k])):
                                                      c.insert(0,i)
                                                      #.extend(comb[k])
                                                      #print c,"c"
                                                      
                                                      c.append(j)
                                                  else:
                                                      c=[comb[k],]
                                                      c.insert(0,i)
                                                      c.append(j)
                                                  
                                                      #add=(i,comb[k],j)
                                                  combod.append(c)
                                                #combod.append(j)
                           #print combod  #this gives [[0, 2, 1], [0, 2, 3, 1], [0, 3, 2, 1], [0, 3, 1]]                                     
                                              #[[3, 0, 2], [3, 0, 1, 2], [3, 1, 0, 2], [3, 1, 2]] for new init rset
                                              
                                    #print combod
                        # make a dictionary like trstops for all paths between transpoints
                           for candidset in range(len(combod)):
                                        #print combod[t]
                                        smlists={}
                                        path6=[]
                                        nextpoints=[]
                                        if len(combod[candidset])==4:#if we are not talking about 2 transf or less
                                            #if comb[k][t]!=comb[k][-1] :#if not last route
                                            #print combod[candidset]
                                            for routeindex in range(len(combod[candidset])-2):
                                                #if not last route in set
                                                    #print routeindex,combod[candidset][routeindex] #gives 0,1
                                                                #start=cur2.index(point)
                                               # if routeindex==0:
                                                 #           cur=routeset[combod[candidset][routeindex]]
                                                  #         start=cur.index(ori)               #print d
                                                    #smlists[combod[candidset][routeindex]]=[]       
                                                #if combod[candidset][routeindex]!=combod[candidset][-1]:
                                                    #make nested list for all strings between routes
                                                    cur3=routeset[combod[candidset][routeindex]] #currently first route
                                                    #print combod[candidset][routeindex+1]
                                                    cur4=routeset[combod[candidset][routeindex+1]]  #currently second route
                                                    if cur3==cur4:continue

                                                    points=trstops[combod[candidset][routeindex]][combod[candidset][routeindex+1]] #transf points between these routes
                                                    #for point in points:
                                                    #print points, cur,cur2
                                                    #if this is the first route
                                                    l=[]
                                                    l1=[]
                                                    #candl=[]
                                                    for point in points:
                                                       if point!=ori and point!=d: 
                                                        if routeindex==0:#ie the origin route
                                                            start1=cur3.index(ori)
                                                            #print ori,start1,cur3,combod[candidset]
                                                            end1=cur3.index(point)  # all possible for origin and trns points
                                                            if start1<end1:
                                                                            l1.append(cur3[start1:end1+1])
                                                            else:
                                                                            lhelp1=cur3[end1:start1+1]
                                                                            lhelp1=lhelp1[::-1]
                                                                            l1.append(lhelp1)
                                                            smlists[combod[candidset][routeindex]]=l1 #for first string
                                                            #print l1,cur3, candidset
                                                        #for interm routes
                                                        start4=cur4.index(point)  # start is each point IN NEXT ROUTE this means i used it to transfer
                                                        #i#f combod[candidset][routeindex]!=combod[candidset][-2]:
                                                        if routeset[combod[candidset][routeindex+1]]==routeset[combod[candidset][routeindex+2]]: continue 

                                                        nextpoints=trstops[combod[candidset][routeindex+1]][combod[candidset][routeindex+2]] 
                                                        for nextpoint in nextpoints:
                                                                if nextpoint!=point and nextpoint!=ori and nextpoint!=d: #they must not be common 
                                                                            end4=cur4.index(nextpoint)
                                                                            if start4<end4:
                                                                                        #l.append(cur2[start:end+1])
                                                                                        candl=(cur4[start4:end4+1])
                                                                            else:
                                                                                        #l.append(cur2[end:start+1])
                                                                                        candl=(cur4[end4:start4+1])
                                                                                        candl=candl[::-1]
                                                                            if candl not in l: l.append(candl)
                                                                            #if routeindex==0: print l                    
                                                                
                                                                            smlists[combod[candidset][routeindex+1]]=l
                                                                            #print cur4,point, nextpoint, candl
                                                                #print smlists
                                                                #if routeindex+1==1: print smlists[combod[candidset][routeindex+1]]
                                            #print smlists
                                            l2=[]
                                            laststr=routeset[combod[candidset][routeindex+2]]#routeset now has value [-2] #OUT OF LOOP
                                            #print laststr,d
                                            #start=cur2.index(point)
                                            end23=laststr.index(d)#else:
                                            if routeset[combod[candidset][routeindex+1]]==routeset[combod[candidset][routeindex+2]]:continue
                                            if nextpoints: 
                                                for np1 in nextpoints:
                                                    if np1!=ori and np1!=d:
                                                        start23=laststr.index(np1)
                                                        if start23<end23:
                                                                                    l2.append(laststr[start23:end23+1])
                                                        else:
                                                                                    l2help=laststr[end23:start23+1]
                                                                                    l2help=l2help[::-1]
                                                                                    l2.append(l2help)
                                                                                    
                                                        smlists[combod[candidset][routeindex+2]]=l2
                                                        
                                                        #print l
                                                        #smlists[combod[candidset][routeindex]]=l
                                                #print smlists
                                                #subfirstp=[]
                                            flag=0
                                            for r in range(len(routeset)):
                                                 #print r,smlists[r]
                                                 if r not in smlists: flag=1
                                            if flag==1: continue
                                            subpaths=[]
                                            times=0
                                            for candroute in range(0,(len(combod[candidset])-1),2):
                                                #for candroute in smlists:
                                                    #store original route index in order of sequence
                                                #print candroute,combod[candidset][candroute]    
                                                times+=1
                                                rind=combod[candidset][candroute]    
                                                posib= smlists[rind] #possible paths on this route
                                                r2ind=combod[candidset][candroute+1]
                                                posib2=smlists[r2ind]
                                                #print rind,r2ind
                                                #print posib,posib2
                                                #subpaths[times]=[]
                                                #search between successive paths
                                               # if combod[candidset][candroute] ==combod[candidset][-1]
                                                for pa1 in posib:
                                                            subpath=[]
                                                            #print pa1,rind
                                                            #for each node sequence for each route
                                                            for nextpath in posib2:#for each poss seq in the next route
                                                                if len(set(pa1).intersection(nextpath))==1: 
                                                                    #print pa1,nextpath, rind,r2ind
                                                                    if pa1[-1]==nextpath[0] :
                                                                        
                                                                #posit=min(p.index(tr[0]),e.index(tr[0]))
                                                                        trans=list(set(pa1).intersection(nextpath))
                                                                        pa1.remove(pa1[-1])
                                                                        subpath=pa1+nextpath
                                                                #print path6, tr, p, e
                                                                    elif pa1[0]==nextpath[-1]:
                                                               # tr=list(set(p).intersection(e))
                                                                #posit=min(p.index(tr[0]),e.index(tr[0]))
                                                                        trans=list(set(pa1).intersection(nextpath))
                                                                        nextpath.remove(nextpath[-1])
                                                                        subpath=nextpath+pa1
                                                                    #print pa1,nextpath,trans
                                                                    #print subpath,rind,r2ind,trans
                                                                    #print pa1,nextpath,subpath
                                                                    if len(subpath)!=0 and subpath not in subpaths:  subpaths.append(subpath)                 
                                            #print subpaths
                                            for p in subpaths:
                                                for e in subpaths:
                                                    if e!=p:
                                                       # print p,e
                                                        if len(set(p).intersection(e))==1:
                                                           # print p,e
                                                            if p[-1]==e[0] :
                                                                tr=list(set(p).intersection(e))
                                                                #posit=min(p.index(tr[0]),e.index(tr[0]))
                                                                #print p,e
                                                                p.remove(p[-1])
                                                                path6=p+e
                                                                #print path6, tr, p, e
                                                            elif e[-1]==p[0]:
                                                                tr=list(set(p).intersection(e))
                                                                #posit=min(p.index(tr[0]),e.index(tr[0]))
                                                                e.remove(e[-1])
                                                                path6=e+p
                                                                #print path6, tr, p, e
                                                                #path6=[]
                                                            sc3=[]
                                                            time=0
                                                            #print "tt", TT[12][13]
                                                            if len(path6)!=0:
                                                                # [ΔΙΟΡΘΩΣΗ 4/4] ιδιο σφαλμα - 3 μετεπιβιβασεις.
                                                                # Η ποινη +15 λεπτων ΔΕΝ αλλαζει.
                                                                time = sum(TT[path6[k]][path6[k+1]] for k in range(len(path6)-1)) if len(path6) >= 2 else 0
                                                                sc3=(time+15,list(path6),3)
                                                                #print path6,sc3
                                                                #print sc[0]
                                                                #if len(paths)!=0: print paths[0][0]
                                                                #do not need to add higher values
                                                                #if len(paths)!=0 and sc1[0]>paths[0][0]:break
                                                                if sc3 not in paths: heapq.heappush(paths,sc3)
                                                            #if path6 not in paths: paths.append(path)
                               #path6.append(subpath)
                                                        #print path6, rind,r2ind, trans
                                                    #print candroute,pospartialpath, combod[candidset][candroute]
                        
                        
    return paths         
            #print paths
            #paths=sorted(paths)
            #print paths
            #ODPATHS[ori][d]= paths           #print smlists                        #              
#print ODPATHS[0][3]            