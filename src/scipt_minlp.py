from pyscipopt import Model, exp, quicksum, quickprod
import autograd.numpy as np
from networkx import topological_sort
from autograd import grad, jacobian
import autograd.numpy as anp # TODO use instead of numpy if autograd is failing


# define problem class
class MRTA_XD():

    def __init__(self, num_tasks, num_robots, dependency_edges, coalition_params, coalition_types, dependency_params,
                 dependency_types,influence_agg_func_types, nodewise_coalition_influence_agg_list,
                 reward_model, task_graph,task_times, makespan_constraint, inter_task_travel_time=None,
                 time_limit=1000, perturbed_objective=False):
        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.dependency_edges = dependency_edges
        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.influence_agg_func_types = influence_agg_func_types
        self.nodewise_coalition_influence_agg_list = nodewise_coalition_influence_agg_list
        self.reward_model = reward_model # need this for the reward model agg functions
        self.task_graph = task_graph
        self.task_times = task_times
        self.makespan_constraint = makespan_constraint
        if inter_task_travel_time is None:
            self.inter_task_travel_time = np.zeros((self.num_tasks, self.num_tasks))
        else:
            self.inter_task_travel_time = inter_task_travel_time
        self.time_limit = time_limit
        self.in_nbrs = []
        for curr_node in range(self.num_tasks):
            self.in_nbrs.append([n for n in self.task_graph.predecessors(curr_node)])

        self.in_edge_inds = []
        self.influence_func_handles = []
        self.perturbed_influence_func_handles = []
        for t in range(self.num_tasks):
            in_edges_t = list(self.task_graph.in_edges(t))
            in_edges_t_inds = [list(self.task_graph.edges).index(edge) for edge in in_edges_t]
            self.in_edge_inds.append(in_edges_t_inds)
            influence_handles_t = [getattr(self, self.reward_model.dependency_types[edge_i]) for edge_i in in_edges_t_inds]
            perturbed_influence_func_handles_t = [getattr(self.reward_model, self.reward_model.dependency_types[edge_i]) for edge_i in in_edges_t_inds]
            self.influence_func_handles.append(influence_handles_t)
            self.perturbed_influence_func_handles.append(perturbed_influence_func_handles_t)

        #self.jacobian_handle = jacobian(self.constraints)

        self.node_order = list(topological_sort(task_graph))

        self.model = Model("MRTA_XD")
        print("MODEL INITIALIZED")
        self.add_variables() # keep variables as lists, but use indexing array
        self.z = self.model.addVar("z")
        self.set_constraints()
        print("CONSTRAINTS SET")
        if not perturbed_objective:
            self.model.addCons(self.z <= self.objective())
            print("CONSTRAINT OBJECTIVE SET")
            self.model.setObjective(self.z, sense='maximize')
            print("OBJECTIVE SET")
        else: # IF PERTURBED OBJECTIVE
            # WE TAKE CARE OF THIS IN
            pass


    def add_variables(self):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
        z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
        s_len = self.num_tasks
        f_len = self.num_tasks
        time_ub = self.time_limit #set this to something sensible at some point
        # add x_ak vars
        self.x_ak = [self.model.addVar("x_%d"%i, vtype="B") for i in range(x_len)] #add binary variables for x_ak
        self.o_akk = [self.model.addVar("o_%d"%i, vtype="B") for i in range(o_len)] #add binary variables for o_akk
        self.z_ak = [self.model.addVar("z_%d"%i,vtype="B") for i in range(z_len)] #add binary variables for z_ak
        self.s_k = [self.model.addVar("s_%d"%i, vtype="C", lb=0, ub=time_ub) for i in range(s_len)] # add continuous vars for s_k
        self.f_k = [self.model.addVar("f_%d"%i, vtype="C", lb=0, ub=time_ub) for i in range(f_len)] # add continuous vars for f_k

        self.ind_x_ak = np.reshape(np.arange(x_len), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a))
        self.ind_o_akk = np.reshape(np.arange(o_len), (self.num_robots, self.num_tasks+1, self.num_tasks))
        # reshape o_akk so that o_akk[a, k+1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. self-edges not included for dummy tasks, but included for all others
        self.ind_z_ak = np.reshape(np.arange(z_len), (self.num_robots, self.num_tasks + 1))


    def objective(self):
        """Returns the scalar value of the objective given x."""

        # x_ak organized by agent
        ind_x_ak = self.ind_x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = [] # np.zeros((self.num_tasks,))
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum([self.x_ak[ind_x_ak[a,t]] for a in range(self.num_robots)])/self.num_robots)
            # the below is just a polynonial or exponential function of the coalition. Can write it explicitly as a separate function in here and then call it
            task_coalition_rewards.append(self.get_coalition(t,task_coalitions[t]))
        #print("coalitions: ", task_coalitions)
        #print("coalition func rewards: ",task_coalition_rewards)
        print("COALITION REWARDS CALCULATED")
        task_influence_rewards = [None for _ in range(self.num_tasks)]
        task_influnce_agg = [None for _ in range(self.num_tasks)]
        task_rewards = [None for _ in range(self.num_tasks)]
        #print(self.influence_func_handles)
        #print(self.in_nbrs)
        task_rewards[0] = 1.0 #SEED TASK HAS ONE REWARD (identity for prod aggregator)
        for t in self.node_order[1:]:
            t = int(t)
            if np.array([task_rewards[self.in_nbrs[t][n]] is None for n in range(len(self.in_nbrs[t]))]).any():
                breakpoint()
            task_influence_rewards[t] = [self.influence_func_handles[t][n](task_rewards[self.in_nbrs[t][n]],self.reward_model.dependency_params[self.in_edge_inds[t][n]]) for n in range(len(self.in_nbrs[t]))]
            if len(self.in_nbrs[t]) == 0:
                task_influence_rewards[t] = [1.0]
            task_influnce_agg[t] = quicksum(task_influence_rewards[t]) # TODO expand this to have more options than just sum
            if self.nodewise_coalition_influence_agg_list[t] == 'sum':
                task_rewards[t] = task_influnce_agg[t] + task_coalition_rewards[t]
            if self.nodewise_coalition_influence_agg_list[t] == 'product':
                task_rewards[t] = task_influnce_agg[t] * task_coalition_rewards[t]
        #print("overall rewards: ", task_rewards)
        print("TASK REWARDS CALCULATED")
        return quicksum(task_rewards[1:]) # - 0.0001*quicksum(self.f_k) # TODO improve upon this super hacky way to incentivize lower times


    def perturbed_objective(self, x, perturbation_type, perturbation_params):
        """Returns the scalar value of the objective given x."""
        assert perturbation_type == 'catastrophic'
        # x_ak organized by agent
        ind_x_ak = self.ind_x_ak[:,1:]
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = [] # np.zeros((self.num_tasks,))
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum([x_ak[ind_x_ak[a,t]] for a in range(self.num_robots)])/self.num_robots)
            # the below is just a polynonial or exponential function of the coalition. Can write it explicitly as a separate function in here and then call it
            task_coalition_rewards.append(self.get_coalition_perturbed(t,task_coalitions[t]))
        #print("coalitions: ", task_coalitions)
        #print("coalition func rewards: ",task_coalition_rewards)
        print("COALITION REWARDS CALCULATED")
        task_influence_rewards = [None for _ in range(self.num_tasks)]
        task_influnce_agg = [None for _ in range(self.num_tasks)]
        task_rewards = [None for _ in range(self.num_tasks)]
        #print(self.influence_func_handles)
        #print(self.in_nbrs)
        task_rewards[0] = 1.0 #SEED TASK HAS ONE REWARD (identity for prod aggregator)
        for t in self.node_order[1:]:
            t = int(t)
            if np.array([task_rewards[self.in_nbrs[t][n]] is None for n in range(len(self.in_nbrs[t]))]).any():
                breakpoint()
            task_influence_rewards[t] = [self.perturbed_influence_func_handles[t][n](task_rewards[self.in_nbrs[t][n]],self.reward_model.dependency_params[self.in_edge_inds[t][n]]) for n in range(len(self.in_nbrs[t]))]
            if len(self.in_nbrs[t]) == 0:
                task_influence_rewards[t] = [1.0]
            task_influnce_agg[t] = np.sum(task_influence_rewards[t]) # TODO expand this to have more options than just sum
            if self.nodewise_coalition_influence_agg_list[t] == 'sum':
                task_rewards[t] = task_influnce_agg[t] + task_coalition_rewards[t]
            if self.nodewise_coalition_influence_agg_list[t] == 'product':
                task_rewards[t] = task_influnce_agg[t] * task_coalition_rewards[t]
            if t in perturbation_params:
                task_rewards[t] = 0.0

        #print("overall rewards: ", task_rewards)
        print("TASK REWARDS CALCULATED")
        return np.sum(task_rewards[1:])

    def init_perturbed_objective_opt(self, perturbation_type, perturbation_params):
        self.perturbation_type = perturbation_type
        self.perturbation_params = perturbation_params
        self.model.addCons(self.z <= self.perturbed_objective_opt())
        print("CONSTRAINT OBJECTIVE SET")
        self.model.setObjective(self.z, sense='maximize')
        print("OBJECTIVE SET")

    def perturbed_objective_opt(self):
        """Returns the scalar value of the objective given x."""
        perturbation_type = self.perturbation_type
        perturbation_params = self.perturbation_params
        assert perturbation_type == 'catastrophic'
        # x_ak organized by agent
        ind_x_ak = self.ind_x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = [] # np.zeros((self.num_tasks,))
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum([self.x_ak[ind_x_ak[a,t]] for a in range(self.num_robots)])/self.num_robots)
            # the below is just a polynonial or exponential function of the coalition. Can write it explicitly as a separate function in here and then call it
            task_coalition_rewards.append(self.get_coalition(t,task_coalitions[t]))
        #print("coalitions: ", task_coalitions)
        #print("coalition func rewards: ",task_coalition_rewards)
        print("COALITION REWARDS CALCULATED")
        task_influence_rewards = [None for _ in range(self.num_tasks)]
        task_influnce_agg = [None for _ in range(self.num_tasks)]
        task_rewards = [None for _ in range(self.num_tasks)]
        #print(self.influence_func_handles)
        #print(self.in_nbrs)
        task_rewards[0] = 1.0 #SEED TASK HAS ONE REWARD (identity for prod aggregator)
        for t in self.node_order[1:]:
            t = int(t)
            if np.array([task_rewards[self.in_nbrs[t][n]] is None for n in range(len(self.in_nbrs[t]))]).any():
                breakpoint()
            task_influence_rewards[t] = [self.influence_func_handles[t][n](task_rewards[self.in_nbrs[t][n]],self.reward_model.dependency_params[self.in_edge_inds[t][n]]) for n in range(len(self.in_nbrs[t]))]
            if len(self.in_nbrs[t]) == 0:
                task_influence_rewards[t] = [1.0]
            task_influnce_agg[t] = quicksum(task_influence_rewards[t]) # TODO expand this to have more options than just sum
            if self.nodewise_coalition_influence_agg_list[t] == 'sum':
                task_rewards[t] = task_influnce_agg[t] + task_coalition_rewards[t]
            if self.nodewise_coalition_influence_agg_list[t] == 'product':
                task_rewards[t] = task_influnce_agg[t] * task_coalition_rewards[t]
            if t in perturbation_params:
                task_rewards[t] = 0.0

        #print("overall rewards: ", task_rewards)
        print("TASK REWARDS CALCULATED")
        return quicksum(task_rewards[1:])

    def set_constraints(self):
        """adds the constraints to the model"""

        # constraint a: every agent starts with one dummy task -- equal to 1
        for a in range(self.num_robots):
            cons_ind = self.ind_x_ak[a,0]
            self.model.addCons(self.x_ak[cons_ind] == 1)

        # constraint d: every task an agent performs has exactly one predecessor
        for a in range(self.num_robots):
            for k_p in range(self.num_tasks):
                cons_inds = self.ind_o_akk[a,:,k_p]
                var_list = [self.o_akk[ind] for ind in cons_inds]
                self.model.addCons(quicksum(var_list) - self.x_ak[self.ind_x_ak[a,k_p+1]] == 0)

        # constraint e: every task an agent performs has exactly one successor except for the last task
        # (including the dummy tasks
        for a in range(self.num_robots):
            for k in range(self.num_tasks+1):
                cons_inds = self.ind_o_akk[a,k,:]
                var_list = [self.o_akk[ind] for ind in cons_inds]
                self.model.addCons(quicksum(var_list) + self.z_ak[self.ind_z_ak[a,k]] - self.x_ak[self.ind_x_ak[a,k]] == 0)

        # constraint f: each robot has exactly one final task
        # TODO is it necessary to multiply by x_ak here??
        for a in range(self.num_robots):
            var_prod_list = [self.x_ak[self.ind_x_ak[a,k]]*self.z_ak[self.ind_z_ak[a,k]] for k in range(self.num_tasks+1)]
            self.model.addCons(quicksum(var_prod_list) == 1)

        # constraint g: eliminate self edges
        for a in range(self.num_robots):
            for k in range(self.num_tasks):
                self.model.addCons(self.o_akk[self.ind_o_akk[a,k+1,k]]==0)

        # PRECEDENCE CONSTRAINTS -- constraint h -- ignoring travel time right now
        for a in range(self.num_robots):
            for k_p in range(len(self.in_nbrs)):
                for k in self.in_nbrs[k_p]:
                    #print("Task ", k ," must precede task ", k_p)
                    #only apply constraint when both tasks are completed by at least one agent
                    is_task_completed_a = quicksum([self.x_ak[self.ind_x_ak[a, k+1]] for a in range(self.num_robots)])
                    is_task_completed_b = quicksum([self.x_ak[self.ind_x_ak[a, k_p+1]] for a in range(self.num_robots)])
                    self.model.addCons(is_task_completed_a*is_task_completed_b*(self.s_k[k_p]-self.f_k[k]-self.inter_task_travel_time[k,k_p]) >= 0)

        # constraint i: time between two consecutive tasks allows for travel time (assumed zero right now)
        for a in range(self.num_robots):
            for k_p in range(self.num_tasks):
                for k in range(self.num_tasks):
                    var = self.o_akk[self.ind_o_akk[a,k+1,k_p]]
                    self.model.addCons(var*(self.s_k[k_p]-self.f_k[k]-self.inter_task_travel_time[k,k_p]) >= 0)

        # DURATION CONSTRAINTS
        for k in range(self.num_tasks):
            self.model.addCons(self.f_k[k] >= self.s_k[k] + self.task_times[k])

        # MAKESPAN CONSTRAINTS
        for k in range(self.num_tasks):
            var_list = [self.x_ak[self.ind_x_ak[a,k+1]] for a in range(self.num_robots)]
            self.model.addCons(quicksum(var_list)*(self.f_k[k]-self.makespan_constraint) <= 0)

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
    def get_coalition_perturbed(self, node_i, f):
        coalition_function = getattr(self.reward_model, self.reward_model.coalition_types[node_i])
        return coalition_function(f, param=self.reward_model.coalition_params[node_i])
    ########################## BELOW FUNCS MODIFIED FROM REWARD_MODEL ##########################
    def sigmoid(self, flow, param):
        return param[0] / (1 + exp(-1 * param[1] * (flow - param[2])))

    def sigmoid_b(self, flow, param):
        return param[0] / (1 + exp(-1 * param[1] * (flow-param[2]))) - param[3]

    def dim_return(self, flow, param):
        if np.array([p is None for p in param]).any() or flow is None:
            breakpoint()
        return param[0] - param[2] * exp(-1 * param[1] * flow)
        # return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    def polynomial(self, flow, param):
        val = quicksum([float(param[i])*flow**i for i in range(len(param))])
        return val

    def exponential(self, flow, param):
        return (param[0]**0.5)*exp(0.5*param[1]*flow)

    def null(self, flow, param):
        return 0.0

    def influence_agg_and(self, deltas):
        return np.prod(np.array(deltas))

    def influence_agg_or(self, deltas):
        #print('or ', deltas, np.sum(np.array(deltas)))
        return np.sum(np.array(deltas))
