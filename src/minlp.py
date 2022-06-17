#import cyipopt
import numpy as np
from autograd import grad, jacobian


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

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)
        # x_ak organized by agent
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        x_dummy = x_ak[:,0]
        x_ak = x_ak[:,1:]
        task_coalitions = np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        task_coalition_rewards = np.zeros((self.num_tasks,))
        import pdb; pdb.set_trace()
        for t in range(self.num_tasks):
            task_coalitions[t] = np.sum(x_ak[:,t])/self.num_robots
            task_coalition_rewards[t] = self.reward_model._compute_node_coalition(t,task_coalitions[t])


        tasks_ordered = np.argsort(np.array(f_k))
        task_rewards = np.zeros((self.num_tasks,))
        for t in tasks_ordered:
            import pdb; pdb.set_trace()
            task_reward, _ = self.reward_model.compute_node_reward_dist(t,task_coalition_rewards[t],[task_rewards[k] for k in self.in_nbrs[t]], np.zeros_like(task_rewards))
            task_rewards[t] = task_reward

        print('task coalitions: ', task_coalitions)
        print('task_rewards: ', task_rewards)

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
        return np.array(self.gradient_func(x))

    def constraints(self, x):
        """Returns the constraints."""
        x_ak, o_akk, z_ak, s_k, f_k = self.partition_x(x)

        # constraint a: every agent starts with one dummy task
        # constraint d: every task has exactly one predecessor
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return np.concatenate((np.prod(x)/x, 2*x))

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

        print(msg.format(iter_count, obj_value))

    def partition_x(self, x):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks - 1)) #extra dummy task, remove duplicates
        z_len = self.num_tasks*self.num_robots #each agent can finish on each task
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
