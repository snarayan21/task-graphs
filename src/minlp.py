import cyipopt
import autograd.numpy as np
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
        self.jacobian_handle = jacobian(self.constraints)

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)
        # x_ak organized by agent
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        x_dummy = x_ak[:,0]
        x_ak = x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = [] # np.zeros((self.num_tasks,))
        for t in range(self.num_tasks):
            try:
                task_coalitions.append(np.sum(x_ak[:,t])/self.num_robots)
            except(ValueError):
                breakpoint()
            task_coalition_rewards.append(self.reward_model._compute_node_coalition(t,task_coalitions[t]))


        tasks_ordered = np.argsort(np.array(f_k))
        task_rewards = [] # np.zeros((self.num_tasks,))
        for t in tasks_ordered:
            task_reward, _ = self.reward_model.compute_node_reward_dist(t,task_coalition_rewards[t],[task_rewards[k] for k in self.in_nbrs[t]], np.zeros_like(task_rewards))
            task_rewards.append(task_reward)

        #print('task coalitions: ', task_coalitions)
        #print('task_rewards: ', task_rewards)
        #print(x)

        return np.sum(task_rewards)

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

    def constraints(self, x):
        """Returns the constraints."""
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        z_ak = np.reshape(np.array(z_ak),(self.num_robots, self.num_tasks + 1))

        # reshape o_akk so that o_akk[a, k-1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. duplicates not included for dummy tasks, but included for all others
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

    # def hessianstructure(self):
    #     """Returns the row and column indices for non-zero vales of the
    #     Hessian."""
    #
    #     # NOTE: The default hessian structure is of a lower triangular matrix,
    #     # therefore this function is redundant. It is included as an example
    #     # for structure callback.
    #
    #     return np.nonzero(np.tril(np.ones((4, 4))))

    # def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""

        # H = obj_factor*np.array((
        #     (2*x[3], 0, 0, 0),
        #     (x[3],   0, 0, 0),
        #     (x[3],   0, 0, 0),
        #     (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
        #
        # H += lagrange[0]*np.array((
        #     (0, 0, 0, 0),
        #     (x[2]*x[3], 0, 0, 0),
        #     (x[1]*x[3], x[0]*x[3], 0, 0),
        #     (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
        #
        # H += lagrange[1]*2*np.eye(4)
        #
        # row, col = self.hessianstructure()
        #
        # return H[row, col]
        # return None

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value), "\n", )

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

#
# lb = [1.0, 1.0, 1.0, 1.0]
# ub = [5.0, 5.0, 5.0, 5.0]
#
# cl = [25.0, 40.0]
# cu = [2.0e19, 40.0]
#
# x0 = [1.0, 5.0, 5.0, 1.0]
#
#
#
#
# nlp = cyipopt.problem(
#    n=len(x0),
#    m=len(cl),
#    problem_obj=MRTA_XD(),
#    lb=lb,
#    ub=ub,
#    cl=cl,
#    cu=cu,
# )
# prob_obj = MRTA_XD()
# #breakpoint()
# nlp.addOption('mu_strategy', 'adaptive')
# nlp.addOption('tol', 1e-7)
#
# x, info = nlp.solve(x0)
# print(x)
