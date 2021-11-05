import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import MathematicalProgramResult
from pydrake.solvers.mathematicalprogram import Solve
from pydrake.symbolic import Expression
#import pydrake.math as math
from pydrake.math import exp
from pydrake.math import atan
#from math import pi
from pydrake.solvers.snopt import SnoptSolver, SnoptSolverDetails
#from math import exp
import matplotlib.pyplot as plt
import networkx as nx
from pydrake.solvers.ipopt import IpoptSolver
from functools import partial

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
        return a/(1+exp(-1*b*(var-c)))
    
    def dimin(a, b, c, var):
        return a+(c*(1-exp(-1*b*var)))

    def stepRho(a, b, c, Ai, fs):
        for i in range(len(Ai)):
            if(Ai[i] == -1):
                Ai[i] = 0

        return a/(1+exp(-1*b*(np.dot(Ai, fs)-c)))

    def stepAtanRho(a, b, c, Ai, fs):
        for i in range(len(Ai)):
            if(Ai[i] == -1):
                Ai[i] = 0

        return (a/2) + a*(atan(b*(np.dot(Ai, fs)-c))/3.14159)
    
    def diminRho(a, b, c, Ai, fs):
        for i in range(len(Ai)):
            if(Ai[i] == -1):
                Ai[i] = 0

        return a+(c*(1-exp(-1*b*np.dot(Ai, fs))))

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
        finalstring = "-1.0*(" + fprodstring + ")*(" + faddstring + ")"
        
        return finalstring, flow_indices

    def parseExpressionGi(strg, strp):
        flow_indices = []
        prodstring = strg + "*" + strp
        addstring = strg + "+" + strp

        for i in range(len(strg)):
            if (strg[i] == 'f'):
                flow_indices.append(int(strg[i+2]))
        
        fprodstring = TaskGraph.replaceVars(prodstring)
        faddstring = TaskGraph.replaceVars(addstring)

        #finalstring = "-1*(" + fprodstring + ")*(" + faddstring + ")"
        finalstring = "-1.0*(" + fprodstring + ")*(" + faddstring + ")"
        
        return finalstring, flow_indices

    def parseExpressionPi(strg, strp):
        flow_indices = []
        prodstring = strg + "*" + strp
        addstring = strg + "+" + strp

        for i in range(len(strp)):
            if (strp[i] == 'f'):
                flow_indices.append(int(strp[i+2]))
        
        fprodstring = TaskGraph.replaceVars(prodstring)
        faddstring = TaskGraph.replaceVars(addstring)

        #finalstring = "-1*(" + fprodstring + ")*(" + faddstring + ")"
        finalstring = "-1.0*(" + fprodstring + ")*(" + faddstring + ")"
        
        return finalstring, flow_indices

    def parseExpressionPiOnly(strp):
        flow_indices = []
        stringy = strp

        for i in range(len(strp)):
            if (strp[i] == 'f'):
                flow_indices.append(int(strp[i+2]))
        
        fstringy = TaskGraph.replaceVars(stringy)

        #finalstring = "-1.0*(" + fprodstring + ")*(" + faddstring + ")"
        finalstring = "-1.0*(" + fstringy + ")"
        
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
        self.f = self.prog.NewContinuousVariables(len(self.edges), "f")
        #p is the coalition component of the reward function for each node 
        self.p = [None]*self.numnodes
        #d is the previous reward component of the reward function for each edge
        self.d = [None]*len(self.edges)
        #r is the reward for each node
        self.r = [None]*self.numnodes
        #c is the combined flow coming into each node
        #self.c = [None]*self.numnodes
        #g is the aggregation of the deltas coming into each node
        self.g = [None]*self.numnodes
        #A is the matrix used for inflows and outflows
        self.A = np.zeros((self.numnodes, len(self.edges)))

        for i in range(len(self.edges)):
            #flow cannot exceed number of robots
            #TODO: do we need this?
            self.prog.AddConstraint(self.f[i] <= self.numrobots)
            #flow cannot be negative
            self.prog.AddConstraint(self.f[i] >= 0)
        
        for i in range(self.numnodes):
            for j in self.edge_dict_rdex[i]:
                #inflow edges for a node i are 1
                self.A[i][j] = 1
            for j in self.edge_dict_ndex[i]:
                #inflow edges for a node i are -1
                self.A[i][j] = -1

        #inflow must equal outflow for nodes that are not source or sink
        self.prog.AddLinearEqualityConstraint(self.A[1:self.numnodes-1], np.zeros(self.numnodes-2), self.f)
        
        #outflow on node 0 (source) must be equal to number of robots
        self.prog.AddLinearEqualityConstraint(-1*self.A[0:1], np.asarray([self.numrobots]), self.f)

        #inflow on last node (sink) must be equal to number of robots
        self.prog.AddLinearEqualityConstraint(self.A[self.numnodes-1:self.numnodes], np.asarray([self.numrobots]), self.f)

        #reward for source node is just a constant -- 1
        self.r[0] = 1
        self.g[0] = 0
        self.p[0] = 1

        #define rho functions as a function of node inflow
        for i in range(1,len(self.rhos)+1):
            rho = self.rhos[i-1]
            rhotype = self.rhotypes[i-1]
            if(rhotype == "s"):
                #self.p[i] = TaskGraph.stepRho(rho[0], rho[1], rho[2], self.A[i].copy(), self.f)
                self.p[i] = TaskGraph.stepAtanRho(rho[0], rho[1], rho[2], self.A[i].copy(), self.f)
                print("node: ", i, "p:", str(self.p[i]))
            else:
                self.p[i] = TaskGraph.diminRho(rho[0], rho[1], rho[2], self.A[i].copy(), self.f)
                print("node: ", i, "p:", str(self.p[i]))

        def addCost(i, fs):
            Ai = self.A[i].copy()
            for k in range(len(Ai)):
                if(Ai[k] == -1):
                    Ai[k] = 0

            rho = self.rhos[i-1]
            rhotype = self.rhotypes[i-1]

            a = rho[0]
            b = rho[1]
            c = rho[2]

            if(rhotype == "s"):
                return -1.0*a/(1+exp(-1*b*(np.dot(Ai, fs)-c)))
            else:
                return -1.0*a+(c*(1-exp(-1*b*np.dot(Ai, fs))))

        #define costs
        for j in range(1,len(self.rhos)+1):
            pfunc = partial(addCost, j)
            print("j is: ", j)
            print(pfunc(self.f))
            c = self.prog.AddCost(pfunc, vars=self.f)
            print(c)
            print("\n")
        
        #self.prog.AddConstraint(self.f[2] >= 4)
        
        print(self.A)

        """ #define delta functions as a function of previous reward
        for i in range(len(self.edges)):
            delta = self.deltas[i]
            deltatype = self.deltatypes[i]
            e = self.edges[i]
            if(deltatype == "s"):
                self.d[i] = TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]])
            else:
                self.d[i] = TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]]) """
        
        """ #define agg functions as functions of incoming deltas
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
            
            #self.r[i] = self.g[i]*self.p[i]*(self.g[i]+self.p[i])
            self.r[i] = self.p[i]

            #self.prog.AddConstraint(self.f[2] >= 3)
            
            #print(str(self.g[i]))
            print(str(self.p[i]))
            #print(type(self.g[i]))
            print(type(self.p[i]))


            if(isinstance(self.g[i], Expression) and isinstance(self.p[i], Expression)):
                fstring, findices = TaskGraph.parseExpression(str(self.g[i]), str(self.p[i]))
                fvars = []
                for k in findices:
                    fvars.append(self.f[k])
                print("fvars:", fvars)
                print("fstring: ", fstring)
                self.prog.AddCost(lambda z: eval(fstring), vars=fvars)
            elif(isinstance(self.g[i], Expression)):
                fstring1, findices1 = TaskGraph.parseExpressionGi(str(self.g[i]), str(self.p[i]))
                fvars1 = []
                for k in findices1:
                    fvars1.append(self.f[k])
                if (len(fvars1) > 0): 
                    print("fstring: ", fstring1)
                    print("fvars:", fvars1)
                    print("g:", str(self.g[i]))
                    print("p:", str(self.p[i]))
                    self.prog.AddCost(lambda z: eval(fstring), vars=fvars1)
            elif(isinstance(self.p[i], Expression)):
                fstring2, findices2 = TaskGraph.parseExpressionPi(str(self.g[i]), str(self.p[i]))
                fvars2 = []
                for k in findices2:
                    fvars2.append(self.f[k])
                if (len(fvars2) > 0): 
                    print("fstring: ", fstring2)
                    print("fvars:", fvars2)
                    print("g:", str(self.g[i]))
                    print("p:", str(self.p[i]))
                    self.prog.AddCost(lambda z: eval(fstring), vars=fvars2)
            
            #print("finished node: ", i, "\n") """
            
    
    def solveGraph(self):
        """ self.prog.SetSolverOption(IpoptSolver().solver_id(), "max_iter", 10000)
        solver = IpoptSolver()
        result = solver.Solve(self.prog) """
        result = Solve(self.prog)
        #print(result.get_solver_details())
        print("Success? ", result.is_success())
        print('f* = ', result.GetSolution(self.f))
        """ print('r* = ', result.GetSolution(self.r))
        print('p* = ', result.GetSolution(self.p))
        print('g* = ', result.GetSolution(self.g))
        print('d* = ', result.GetSolution(self.d))
        print('c* = ', result.GetSolution(self.c)) """
        print('optimal cost = ', result.get_optimal_cost())
        print('solver is: ', result.get_solver_id().name())
        aa = result.get_solver_details().info
        print('Solver Status: ', aa)


