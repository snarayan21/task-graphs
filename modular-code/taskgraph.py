import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
from pydrake.symbolic import Expression
import pydrake.math as math
from math import exp
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

    def replaceVars(string):
        lstring = list(string)
        i = 0
        j = 0
        while i < len(lstring):
            if (lstring[i] == 'f'):
                lstring[i] = 'z'
                i += 1
                lstring[i] = '['
                i += 1
                lstring[i] = str(j)
                j += 1
                i += 1
                lstring[i] = ']'
            i += 1
        
        return "".join(lstring)


    def parseExpression(strg, strp):
        flow_indices = []
        prodstring = strg + "*" + strp
        addstring = strg + "+" + strp

        for i in range(len(strg)):
            if (strg[i] == 'f'):
                flow_indices.append(int(strg[i+2]))
        
        for i in range(len(strp)):
            if (strp[i] == 'f'):
                flow_indices.append(int(strp[i+2]))
        
        fprodstring = TaskGraph.replaceVars(prodstring)
        faddstring = TaskGraph.replaceVars(addstring)

        #finalstring = "-1*(" + fprodstring + ")*(" + faddstring + ")"
        finalstring = "-1*(" + fprodstring + ")*(" + faddstring + ")"
        
        return finalstring, flow_indices
        

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

        #f is the flow across each edge
        self.f = self.prog.NewContinuousVariables(len(self.edges_tot), "f")
        #p is the coalition component of the reward function for each node 
        self.p = [None]*self.numnodes
        #d is the previous reward component of the reward function for each edge
        self.d = [None]*len(self.edges)
        #r is the reward for each node
        self.r = [None]*self.numnodes
        #c is the combined flow coming into each node
        self.c = [None]*self.numnodes
        #g is the aggregation of the deltas coming into each node
        self.g = [None]*self.numnodes

        #TODO: WRITE CONSTRAINTS WHOLLY IN TERMS OF f VARIABLES

        #all these variables must be positive
        #TODO: THESE MAY NOT BE NEEDED!
        """ for i in range(self.numnodes):
            self.prog.AddConstraint(self.g[i] >= 0)
            self.prog.AddConstraint(self.c[i] >= 0)
            self.prog.AddConstraint(self.r[i] >= 0)
            self.prog.AddConstraint(self.p[i] >= 0) """

        #TODO: THIS MAY NOT BE NEEDED!
        """ for i in range(len(self.edges)):
            self.prog.AddConstraint(self.d[i] >= 0) """

        for i in range(len(self.edges)):
            #flow cannot exceed number of robots
            self.prog.AddConstraint(self.f[i] <= self.numrobots)
            #flow cannot be negative
            self.prog.AddConstraint(self.f[i] >= 0)
            #flow over normal edges is inverse of flow on reverse edges
            #TODO: THIS MAY NOT BE NEEDED!
            #self.prog.AddConstraint(self.f[i] == -1*self.f[i+len(self.edges)])
        
        #set the inflow of source node to 0
        self.c[0] = 0

        for i in range(1,self.numnodes):
            inflow = 0
            for j in self.edge_dict_rdex[i]:
                inflow = inflow + self.f[j]

            #c[i] is the inflow to node i -- important for rho function
            self.c[i] = inflow
        
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
        self.c[self.numnodes-1] = self.numrobots

        #reward for source node is just a constant -- 1
        #TODO: SINCE THESE ARE NOT DECISION VARIABLES YOU CAN JUST MAKE r[0] A CONSTANT
        self.r[0] = 1
        self.g[0] = 0
        self.p[0] = 1

        #define rho functions as a function of node inflow
        for i in range(1,len(self.rhos)+1):
            rho = self.rhos[i-1]
            rhotype = self.rhotypes[i-1]
            if(rhotype == "s"):
                self.p[i] = TaskGraph.step(rho[0], rho[1], rho[2], self.c[i])
            else:
                self.p[i] = TaskGraph.dimin(rho[0], rho[1], rho[2], self.c[i])
        
        """ #define delta functions as a function of previous reward
        for i in range(len(self.edges)):
            delta = self.deltas[i]
            deltatype = self.deltatypes[i]
            e = self.edges[i]
            if(deltatype == "s"):
                self.d[i] = TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]])
            else:
                self.d[i] = TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]]) """
        
        #define agg functions as functions of incoming deltas
        for i in range(1,self.numnodes):
            agg = self.aggs[i-1]
            indeltas = []
            for j in self.edge_dict_rdex[i]:
                e = self.edges[j]
                print("edge:", e)
                delta = self.deltas[j]
                deltatype = self.deltatypes[j]
                if(deltatype == "s"):
                    self.d[j] = TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]])
                else:
                    self.d[j] = TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]])
                indeltas.append(self.d[j])
            
            indeltas = np.array(indeltas)
            
            if(agg == "a"):
                self.g[i] = TaskGraph.add(indeltas)
            
            elif(agg == "m"):
                self.g[i] = TaskGraph.mult(indeltas)

            else:
                self.g[i] = TaskGraph.combo(indeltas)
            
            self.r[i] = self.g[i]*self.p[i]*(self.g[i]+self.p[i])

            #self.prog.AddConstraint(self.f[1] >= 2)
            
            print(str(self.g[i]))
            print(str(self.p[i]))
            print(type(self.g[i]))
            print(type(self.p[i]))
            if(isinstance(self.g[i], Expression) and isinstance(self.p[i], Expression)):
                fstring, findices = TaskGraph.parseExpression(str(self.g[i]), str(self.p[i]))
                fvars = []
                for k in findices:
                    fvars.append(self.f[k])
                print("fvars:", fvars)
                print("fstring: ", fstring)
                self.prog.AddCost(lambda z: eval(fstring), vars=fvars)
            else:
                self.prog.AddCost(-1*self.r[i])
            
            print("finished node: ", i)
            
    
    def solveGraph(self):
        result = Solve(self.prog)
        #print(result.getSolverDetails())
        print("Success? ", result.is_success())
        print('f* = ', result.GetSolution(self.f))
        """ print('r* = ', result.GetSolution(self.r))
        print('p* = ', result.GetSolution(self.p))
        print('g* = ', result.GetSolution(self.g))
        print('d* = ', result.GetSolution(self.d))
        print('c* = ', result.GetSolution(self.c)) """
        print('optimal cost = ', result.get_optimal_cost())
        print('solver is: ', result.get_solver_id().name())


