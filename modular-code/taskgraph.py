import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
import pydrake.math as math
import matplotlib.pyplot as plt
import networkx as nx

class TaskGraph():
    #class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self, numnodes, edges, rhos, rhotypes, deltas, deltatypes, aggs, numrobots):
        self.numnodes = numnodes
        self.edges = edges
        self.rhos = rhos
        self.rhotypes = rhotypes
        self.deltas = deltas
        self.deltatypes = deltatypes
        self.aggs = aggs
        self.numrobots = numrobots

    def step(a, b, c, var):
        return a/(1+np.exp(-1*b*(var-c)))
    
    def dimin(a, b, c, var):
        return a+(c*(1-np.exp(-1*b*var)))

    def mult(vars):
        return np.prod(vars)

    def add(vars):
        return np.sum(vars)

    def combo(vars):
        return np.prod(vars)*np.sum(vars)

    def initializeSolver(self):
        #this function will define variables, functions, and bounds based on the input info
        self.prog = MathematicalProgram()

        #edges_tot will include all edges and their reverses
        self.edges_tot = self.edges.copy()

        for e in self.edges:
            self.edges_tot.append([e[1], e[0]])
        
        self.edge_dict_n = {}
        self.edge_dict_r = {}
        self.edge_dict_ndex = {}
        self.edge_dict_rdex = {}

        for i in range(len(self.edges)):
            e = self.edges[i]
            if e[0] in self.edge_dict_n:
                self.edge_dict_n[e[0]].append(e[1])
            else:
                self.edge_dict_n[e[0]] = [e[1]]
            
            if e[0] in self.edge_dict_ndex:
                self.edge_dict_ndex[e[0]].append(i)
            else:
                self.edge_dict_ndex[e[0]] = [i]

            if e[1] in self.edge_dict_r:
                self.edge_dict_r[e[1]].append(e[0])
            else:
                self.edge_dict_r[e[1]] = [e[0]]
            
            if e[1] in self.edge_dict_rdex:
                self.edge_dict_rdex[e[1]].append(i)
            else:
                self.edge_dict_rdex[e[1]] = [i]

        for i in range(self.numnodes):
            if i not in self.edge_dict_n:
                self.edge_dict_n[i] = []
            if i not in self.edge_dict_r:
                self.edge_dict_r[i] = []
            if i not in self.edge_dict_ndex:
                self.edge_dict_ndex[i] = []
            if i not in self.edge_dict_rdex:
                self.edge_dict_rdex[i] = []

        print(self.edge_dict_ndex)
        print(self.edge_dict_rdex)

        #f is the flow across each edge
        self.f = self.prog.NewContinuousVariables(len(self.edges_tot), "f")
        #p is the coalition component of the reward function for each node 
        self.p = self.prog.NewContinuousVariables(self.numnodes, "p")
        #d is the previous reward component of the reward function for each edge
        self.d = self.prog.NewContinuousVariables(len(self.edges), "d")
        #r is the reward for each node
        self.r = self.prog.NewContinuousVariables(self.numnodes, "r")
        #c is the combined flow coming into each node
        self.c = self.prog.NewContinuousVariables(self.numnodes, "c")
        #g is the aggregation of the deltas coming into each node
        self.g = self.prog.NewContinuousVariables(self.numnodes, "g")

        #all these variables must be positive
        for i in range(self.numnodes):
            self.prog.AddConstraint(self.g[i] >= 0)
            self.prog.AddConstraint(self.c[i] >= 0)
            self.prog.AddConstraint(self.r[i] >= 0)
            self.prog.AddConstraint(self.p[i] >= 0)

        for i in range(len(self.edges)):
            self.prog.AddConstraint(self.d[i] >= 0)

        for i in range(len(self.edges)):
            #flow cannot exceed number of robots
            self.prog.AddConstraint(self.f[i] <= self.numrobots)
            #flow over normal edges is inverse of flow on reverse edges
            self.prog.AddConstraint(self.f[i] == -1*self.f[i+len(self.edges)])
        
        for i in range(self.numnodes):
            inflow = []
            for j in self.edge_dict_rdex[i]:
                inflow.append(self.f[j])

            inflow = np.array(inflow)

            #c[i] is the inflow to node i -- important for rho function
            self.prog.AddConstraint(self.c[i] == np.sum(inflow))
        
        #set the inflow of source node to 0
        self.prog.AddConstraint(self.c[0] == 0)

        for i in range(1,self.numnodes-1):
            outflow = []
            for j in self.edge_dict_ndex[i]:
                outflow.append(self.f[j])
            
            outflow = np.array(outflow)

            #c[i], which is node inflow, must be equal to node outflow (flow conservation)
            #this does not apply to the source or the sink
            self.prog.AddConstraint(self.c[i] - np.sum(outflow) == 0)
        
        #outflow on node 0 (source) must be equal to number of robots
        source_outflow = []
        for i in self.edge_dict_ndex[0]:
            source_outflow.append(self.f[i])
        source_outflow = np.array(source_outflow)
        self.prog.AddConstraint(np.sum(source_outflow) == self.numrobots)

        #inflow on last node (sink) must be equal to number of robots
        self.prog.AddConstraint(self.c[self.numnodes-1] == self.numrobots)

        #reward for source node is just a constant -- 1
        self.prog.AddConstraint(self.p[0] == 1)
        self.prog.AddConstraint(self.g[0] == 0)
        self.prog.AddConstraint(self.r[0] == 1)

        #define rho functions as a function of node inflow
        for i in range(1,len(self.rhos)+1):
            rho = self.rhos[i-1]
            rhotype = self.rhotypes[i-1]
            if(rhotype == "s"):
                print("node", i, ":", *rho)
                self.prog.AddConstraint(self.p[i] == TaskGraph.step(rho[0], rho[1], rho[2], self.c[i]))
            else:
                self.prog.AddConstraint(self.p[i] == TaskGraph.dimin(rho[0], rho[1], rho[2], self.c[i]))
        
        #define delta functions as a function of previous reward
        for i in range(len(self.edges)):
            delta = self.deltas[i]
            deltatype = self.deltatypes[i]
            e = self.edges[i]
            if(deltatype == "s"):
                self.prog.AddConstraint(self.d[i] == TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]]))
            else:
                self.prog.AddConstraint(self.d[i] == TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]]))
        
        #define agg functions as functions of incoming deltas
        for i in range(1,self.numnodes):
            agg = self.aggs[i-1]
            indeltas = []
            for j in self.edge_dict_rdex[i]:
                indeltas.append(self.d[j])
            
            indeltas = np.array(indeltas)
            
            if(agg == "a"):
                self.prog.AddConstraint(self.g[i] == TaskGraph.add(indeltas))
            
            elif(agg == "m"):
                self.prog.AddConstraint(self.g[i] == TaskGraph.mult(indeltas))

            else:
                self.prog.AddConstraint(self.g[i] == TaskGraph.combo(indeltas))

        #define reward as "combo" of rho and agg
        for i in range(1,self.numnodes):
            self.prog.AddConstraint(self.r[i] == self.g[i]*self.p[i]*(self.g[i]+self.p[i]))
            #here we make the sign of reward negative since the solver is minimizing
            #which is equivalent to maximizing the positive reward
            self.prog.AddCost(-1*self.r[i])
    
    def solveGraph(self):
        result = Solve(self.prog)
        print("Success? ", result.is_success())
        print('f* = ', result.GetSolution(self.f))
        print('r* = ', result.GetSolution(self.r))
        print('p* = ', result.GetSolution(self.p))
        print('g* = ', result.GetSolution(self.g))
        print('d* = ', result.GetSolution(self.d))
        print('c* = ', result.GetSolution(self.c))
        print('optimal cost = ', result.get_optimal_cost())
        print('solver is: ', result.get_solver_id().name())
            

















