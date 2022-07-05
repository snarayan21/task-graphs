import cyipopt
from pyscipopt import Model, exp, quicksum
import autograd.numpy as np
from networkx import topological_sort
from autograd import grad, jacobian
import autograd.numpy as anp # TODO use instead of numpy if autograd is failing


# define problem class
class MRTA_XD():

    def __init__(self, num_tasks, num_robots, dependency_edges, coalition_params, coalition_types, dependency_params,
                 dependency_types,influence_agg_func_types, reward_model, task_graph):
        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.dependency_edges = dependency_edges
        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.influence_agg_func_types = influence_agg_func_types
        self.reward_model = reward_model # need this for the reward model agg functions
        self.gradient_func = None
        self.task_graph = task_graph
        self.in_nbrs = []
        for curr_node in range(self.num_tasks):
            self.in_nbrs.append([n for n in self.task_graph.predecessors(curr_node)])

        self.in_edge_inds = []
        self.influence_func_handles = []
        for t in range(self.num_tasks):
            in_edges_t = list(self.task_graph.in_edges(t))
            in_edges_t_inds = [list(self.task_graph.edges).index(edge) for edge in in_edges_t]
            self.in_edge_inds.append(in_edges_t_inds)
            influence_handles_t = [getattr(self, self.reward_model.dependency_types[edge_i]) for edge_i in in_edges_t_inds]
            self.influence_func_handles.append(influence_handles_t)

        #self.jacobian_handle = jacobian(self.constraints)

        self.node_order = list(topological_sort(task_graph))

        self.model = Model("MRTA_XD")
        self.add_variables() # keep variables as lists, but use indexing array
        #self.model.setObjective(self.objective)

        #self.set_constraints()

    def add_variables(self):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
        z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
        s_len = self.num_tasks
        f_len = self.num_tasks
        time_ub = 1000 #set this to something sensible at some point
        # add x_ak vars
        self.x_ak = [self.model.addVar("", vtype="B") for _ in range(x_len)] #add binary variables for x_ak
        self.o_akk = [self.model.addVar("", vtype="B") for _ in range(o_len)] #add binary variables for o_akk
        self.z_ak = [self.model.addVar("",vtype="B") for _ in range(z_len)] #add binary variables for z_ak
        self.s_k = [self.model.addVar("", vtype="C", lb=0, ub=1000) for _ in range(s_len)] # add continuous vars for s_k
        self.f_k = [self.model.addVar("", vtype="C", lb=0, ub=1000) for _ in range(f_len)] # add continuous vars for f_k

        self.ind_x_ak = np.reshape(np.arange(x_len), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a))
        self.ind_o_akk = np.reshape(np.arange(o_len), (self.num_robots, self.num_tasks+1, self.num_tasks))
        # reshape o_akk so that o_akk[a, k-1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. self-edges not included for dummy tasks, but included for all others
        self.ind_z_ak = np.reshape(np.arange(z_len), (self.num_robots, self.num_tasks + 1))


    def objective(self, x):
        #may have to make the member variables argument variables instead, so the objective func takes in arguments
        """Returns the scalar value of the objective given x."""
        self.x_ak, self.o_akk, self.z_ak, self.s_k, self.f_k = self.partition_x(x) #uncomment for testing with values
        # x_ak organized by agent
        ind_x_dummy = self.ind_x_ak[:,0]
        ind_x_ak = self.ind_x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = [] # np.zeros((self.num_tasks,))
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum([self.x_ak[ind_x_ak[a,t]] for a in range(self.num_robots)])/self.num_robots)
            # the below is just a polynonial or exponential function of the coalition. Can write it explicitly as a separate function in here and then call it
            task_coalition_rewards.append(self.get_coalition(t,task_coalitions[t]))
        print("coalitions: ", task_coalitions)
        print("coalition func rewards: ",task_coalition_rewards)

        task_influence_rewards = [None for _ in range(self.num_tasks)]
        task_influnce_agg = [None for _ in range(self.num_tasks)]
        task_rewards = [None for _ in range(self.num_tasks)] # np.zeros((self.num_tasks,))
        print(self.influence_func_handles)
        print(self.in_nbrs)
        for t in self.node_order:
            task_influence_rewards[t] = [self.influence_func_handles[t][n](task_rewards[self.in_nbrs[t][n]],self.reward_model.dependency_params[self.in_edge_inds[t][n]]) for n in range(len(self.in_nbrs[t]))]
            task_influnce_agg[t] = quicksum(task_influence_rewards[t]) # TODO expand this to have more options than just sum
            task_rewards[t] = task_influnce_agg[t] + task_coalition_rewards[t] # TODO expand this to have more options than just sum
        print("overall rewards: ", task_rewards)

        """

        for t in tasks_ordered:
            task_reward, _ = self.reward_model.compute_node_reward_dist(t,task_coalition_rewards[t],[task_rewards[k] for k in self.in_nbrs[t]], np.zeros_like(task_rewards))
            task_rewards.append(task_reward)
        """
        #print('task coalitions: ', task_coalitions)
        #print('task_rewards: ', task_rewards)
        #print(x)

        return quicksum(task_rewards)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        # return np.array([
        #     x[0]*x[3] + x[3]*np.sum(x[0:3]),
        #     x[0]*x[3],
        #     x[0]*x[3] + 1.0,
        #     x[0]*np.sum(x[0:3])
        # ])
        if self.gradient_func == None:
            self.gradient_func = grad(self.objective)
        #print(np.array(self.gradient_func(x)))
        return np.array(self.gradient_func(x))

    def set_constraints(self, x):
        """Returns the constraints."""
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        z_ak = np.reshape(np.array(z_ak), (self.num_robots, self.num_tasks + 1))

        # reshape o_akk so that o_akk[a, k-1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. self-edges not included for dummy tasks, but included for all others
        o_akk = np.reshape(np.array(o_akk), (self.num_robots, self.num_tasks+1, self.num_tasks))

        constraints = []

        # constraint a: every agent starts with one dummy task -- equal to 1
        for a in range(self.num_robots):
            constraints.append(x_ak[a,0] - 1)

        # constraint d: every task an agent performs has exactly one predecessor
        for a in range(self.num_robots):
            for k_p in range(self.num_tasks):
                constraints.append(np.sum(o_akk[a,:,k_p])-x_ak[a,k_p+1])
        # constraint e: every task an agent performs has exactly one successor except the last task
        for a in range(self.num_robots):
            for k in range(self.num_tasks+1): #K_a+
                constraints.append(np.sum(o_akk[a,k,:]) + z_ak[a,k] - x_ak[a,k])

        return np.array(constraints)

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return np.array(self.jacobian_handle(x))

    def partition_x(self, x):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
        z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
        s_len = self.num_tasks
        f_len = self.num_tasks
        x_ak = x[0:x_len]
        o_akk = x[x_len:x_len + o_len]
        z_ak = x[x_len + o_len : x_len + o_len + z_len]
        s_k = x[x_len + o_len + z_len : x_len + o_len + z_len + s_len ]
        f_k = x[x_len + o_len + z_len + s_len : x_len + o_len + z_len + s_len + f_len]

        return x_ak, o_akk, z_ak, s_k, f_k

    def get_coalition(self, node_i, f):

        coalition_function = getattr(self, self.reward_model.coalition_types[node_i])
        #breakpoint()
        return coalition_function(f, param=self.reward_model.coalition_params[node_i])

    ########################## BELOW FUNCS MODIFIED FROM REWARD_MODEL ##########################
    def sigmoid(self, flow, param):
        return param[0] / (1 + exp(-1 * param[1] * (flow - param[2])))

    def dim_return(self, flow, param):
        return param[0] - param[2] * exp(-1 * param[1] * flow)
        # return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    def polynomial(self, flow, param):
        val = quicksum([float(param[i])*flow**i for i in range(len(param))])
        return val

    def influence_agg_and(self, deltas):
        return np.prod(np.array(deltas))

    def influence_agg_or(self, deltas):
        #print('or ', deltas, np.sum(np.array(deltas)))
        return np.sum(np.array(deltas))
