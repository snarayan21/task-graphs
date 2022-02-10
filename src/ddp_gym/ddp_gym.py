#import gym
#import env
from autograd import grad, jacobian
import autograd.numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers



"""
saaketh's important git commands
git merge origin/main
git push origin HEAD:ddp-integration-qp
"""

class DDP:
    def __init__(self, next_state, #vector of dynamics equations function handles -- one for each node
                 running_cost,  #vector of running cost function handles -- one for each node
                 final_cost, #function of final cost
                 umax, 
                 state_dim, 
                 pred_time,
                 inc_mat,
                 adj_mat,
                 edgelist,
                 constraint_type='qp'):
        self.pred_time = pred_time
        self.umax = umax
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.l = running_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.incmat = inc_mat
        self.adjmat = adj_mat
        self.edgelist = edgelist
        self.constraint_type = constraint_type


    def backward(self, x_seq, u_seq):
        # initialize value func
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        #TODO: make an incoming_x_seq that places the list of incoming node rewards to node l at index l
        incoming_x_seq = self.x_seq_to_incoming_x_seq(x_seq)
        incoming_u_seq = self.u_seq_to_incoming_u_seq(u_seq)
        #breakpoint()
        incoming_node_list = self.get_incoming_node_list() #gives the order of the lists of incoming nodes to each node
        k_seq = []
        kk_seq = []
        for l in range(self.pred_time - 1, -1, -1): # (num_tasks-2, num_tasks-3, ..., 0)
            incoming_rewards_arr = list(incoming_x_seq[l])
            if l in incoming_node_list[l]:
                l_ind = incoming_node_list[l].index(l)
                x = incoming_rewards_arr[l_ind]
                incoming_rewards_arr.pop(l_ind)
                additional_x = incoming_rewards_arr
            else:
                l_ind = -1
                additional_x = incoming_rewards_arr
                x = np.array([[0.0]])

            l_x = grad(self.l[l], 0)
            l_u = grad(self.l[l], 1)
            l_xx = jacobian(l_x, 0)
            l_uu = jacobian(l_u, 1)
            l_ux = jacobian(l_u, 0)
            f_x = jacobian(self.f[l], 0)
            f_u = jacobian(self.f[l], 1)
            f_xx = jacobian(f_x, 0)
            f_uu = jacobian(f_u, 1)
            f_ux = jacobian(f_u, 0)
            #print(l)
            #breakpoint()
            f_x_t = f_x(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind)
            f_u_t = f_u(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind)
            #breakpoint()
            q_x = l_x(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind) + np.matmul(np.atleast_2d(f_x_t).T, np.atleast_2d(self.v_x[l + 1]))
            q_u = l_u(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind) + np.squeeze(np.matmul(np.atleast_2d(f_u_t).T, np.atleast_2d(self.v_x[l + 1])))
            #breakpoint()
            q_xx = l_xx(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind) + \
              np.matmul(np.atleast_2d(np.matmul(np.atleast_2d(f_x_t).T, np.atleast_2d(self.v_xx[l + 1]))), np.atleast_2d(f_x_t)) + \
              np.dot(np.atleast_1d(self.v_x[l + 1]), np.atleast_1d(np.squeeze(f_xx(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind))))
            tmp = np.matmul(np.atleast_2d(f_u_t).T, np.atleast_2d(self.v_xx[l + 1]))
            q_uu = l_uu(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind) + np.matmul(np.atleast_2d(tmp), np.atleast_2d(f_u_t)) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_uu(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind)))
            #q_uu = np.array([[1.0]])

            #for some reason q_uu has 4 dimensions sometimes.
            if(np.size(q_uu) == 1):
                q_uu = np.atleast_2d(np.squeeze(q_uu))

            if not np.all(np.linalg.eigvals(q_uu) > 0):
                lam = -np.min(np.linalg.eigvals(q_uu))
                if q_uu.shape[0] == 1:
                    q_uu = -q_uu  #TODO IS THIS REGULARIZATION????
                else:
                    q_uu = np.eye(q_uu.shape[0])*1.0 #TODO IS THIS REGULARIZATION????

                print("regularizing Quu with lambda = ", lam)
            q_ux = np.squeeze(l_ux(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind)) + np.squeeze(np.matmul(np.atleast_1d(tmp), np.atleast_1d(f_x_t))) + \
              np.squeeze(np.dot(self.v_x[l + 1], np.squeeze(f_ux(np.atleast_1d(x), np.atleast_1d(incoming_u_seq[l]), np.atleast_1d(additional_x), l_ind))))

            
            try:
                inv_q_uu = np.linalg.inv(np.atleast_2d(q_uu))
            except np.linalg.LinAlgError:
                print("got to here wtf")
                inv_q_uu = np.zeros_like(q_uu)
                print('SINGULAR MATRIX: RETURNING ZERO GRADIENT')

            q_uu = np.atleast_2d(q_uu)
            q_x = np.atleast_2d(q_x)

            print('q_x: ', q_x)
            print('q_u: ', q_u)
            print('q_ux: ', q_ux)
            print('q_uu: ', q_uu)
            print('incoming u seq: ', incoming_u_seq[l])
            
            
            if self.constraint_type == 'qp':

                opts = {'reltol' : 1e-10, 'abstol' : 1e-10, 'feastol' : 1e-10}

                nu = q_u.size
                print(nu)
                curr_inc_mat = self.incmat[l+1]
                curr_u = []
                #curr node inflow
                u = 0.0
                for i in range(len(curr_inc_mat)):
                    if(curr_inc_mat[i] == 1):
                        u += u_seq[i]
                        curr_u.append(u_seq[i])
                #curr node outflow
                p = 0.0
                for i in range(len(curr_inc_mat)):
                    if(curr_inc_mat[i] == -1):
                        p += u_seq[i]
                
                #curr shared inflow (from  other edges)
                """ s = 0.0
                for i in range(len(curr_inc_mat)):
                    if(curr_inc_mat[i] == -1):
                        s += u_seq[i] """

                solns = np.zeros(nu)
                
                if(l == self.pred_time - 1):
                    #constraint on p doesn't take place if we are at last node. Only z slack variable
                    P = np.copy(q_uu)
                    #P = np.hstack((P, np.zeros((P.shape[0], 2))))
                    #P = np.vstack((P, np.zeros((2, P.shape[1]))))
                    #P = cvxopt_matrix(P, tc='d')
                    #q = np.copy(q_x)
                    q = np.copy(q_u)
                    #q = np.vstack((q, np.zeros((2,1))))
                    #A = np.ones((1, nu+1))
                    A = np.full((2, nu+2), 1)
                    #u + Adu <= 1
                    #Adu <= 1-u
                    #Adu + z = 1-u
                    #z >= 0
                    A[0][-2] = 1
                    A[0][-1] = 0
                    #u + Adu >= 0
                    #Adu >= -u
                    #Adu + x = -u
                    #x <= 0
                    A[1][-2] = 0
                    A[1][-1] = 1
                    #b = np.array([1 - u])
                    b = np.array([1-u, -u])
                    #G = np.zeros((1, nu+1))
                    G = np.zeros(((2*nu) + 2, nu))
                    #G[0][-2] = -1
                    #G[1][-1] = 1
                    G[0] = np.ones(nu)
                    G[1] = -1*np.ones(nu)
                    #(1) u_i + du_i >= 0 --> du_i >= -u_i --> -du_i <= u_i
                    #and
                    #(2) u_i + du_i <= 1 --> du_i <= 1 - u_i
                    for c in range(nu):
                        #for constr (1)
                        G[c+2][c] = -1
                        #for constr (2)
                        G[c+nu+2][c] = 1
                    h = np.zeros(((2*nu) + 2, 1))
                    #u+ Adu <= 1 --> Adu <= 1-u
                    h[0] = 1-u
                    #u + Adu >= 0 --> Adu >= -u --> -Adu <= u
                    h[1] = u
                    for i in range(nu):
                        #use curr_u vector from above
                        #TODO: confirm that indices are consistent
                        h[i+2] = curr_u[i]
                    for i in range(nu, 2*nu):
                        #use curr_u vector from above
                        #TODO: confirm that indices are consistent
                        h[i+2] = 1 - curr_u[i-nu]

                    P = cvxopt_matrix(P, tc='d')
                    q = cvxopt_matrix(q, tc='d')
                    A = cvxopt_matrix(A, tc='d')
                    b = cvxopt_matrix(b, tc='d')
                    G = cvxopt_matrix(G, tc='d')
                    h= cvxopt_matrix(h, tc='d')
                    
                    #soln = cvxopt_solvers.qp(P, q, G, h, A, b)
                    print("we are in the edge case!")
                    print("P", P)
                    print("q", q)
                    print("G", G)
                    print("h", h)
                    print("u", u)
                    print("p", p)
                    #breakpoint()
                    soln = cvxopt_solvers.qp(P, q, G, h, options = opts)
                    sols = np.array(soln['x']).reshape(1,-1)[0]
                    print("SOLUTION: ", sols)
                    #breakpoint()
                    solns = sols

                else:
                    P = np.copy(q_uu)
                    #P = np.hstack((P, np.zeros((P.shape[0], 3))))
                    #P = np.vstack((P, np.zeros((3, P.shape[1]))))
                    #q = np.copy(q_x)
                    q = np.copy(q_u)
                    #q = np.vstack((q, np.zeros((3,1)))) 
                    A = np.full((3, nu+3), 1)
                    #u + Adu >= p
                    #Adu >= p-u
                    #Adu + y = p-u
                    #y <= 0
                    A[0][-3] = 1
                    A[0][-2] = 0
                    A[0][-1] = 0
                    #u + Adu <= 1
                    #Adu <= 1-u
                    #Adu + z = 1-u
                    #z >= 0
                    A[1][-3] = 0
                    A[1][-2] = 1
                    A[1][-1] = 0
                    #u + Adu >= 0
                    #Adu >= -u
                    #Adu + x = -u
                    #x <= 0
                    A[2][-3] = 0
                    A[2][-2] = 0
                    A[2][-1] = 1
                    print(A)
                    b = np.array([p-u, 1-u, -u])
                    #need to add constraints on inflow components and on slack variables
                    #G = np.zeros(((2*nu) + 3, nu+3))
                    G = np.zeros(((2*nu) + 4, nu))
                    #y <= 0
                    #G[0][-3] = 1
                    #z >= 0 --> -z <= 0
                    #G[1][-2] = -1
                    #x <= 0
                    #G[2][-1] = 1
                    G[0] = np.ones(nu)
                    G[1] = -1*np.ones(nu)
                    G[2] = -1*np.ones(nu)
                    G[3] = np.ones(nu)
                    #(1) u_i + du_i >= 0 --> du_i >= -u_i --> -du_i <= u_i
                    #and
                    #(2) u_i + du_i <= 1 --> du_i <= 1 - u_i
                    for c in range(nu):
                        #for constr (1)
                        G[c+4][c] = -1
                        #for constr (2)
                        G[c+nu+4][c] = 1
                    h = np.zeros(((2*nu) + 4, 1))
                    #u+ Adu <= 1 --> Adu <= 1-u
                    h[0] = 1-u
                    #u + Adu >= 0 --> Adu >= -u --> -Adu <= u
                    h[1] = u
                    #u + Adu >= p --> Adu >= p - u --> -Adu <= u - p
                    h[2] = u-p
                    h[3] = p-u
                    #breakpoint()
                    if(p > 1.0):
                        h[2] = u-1.0
                        h[3] = 1.0-u
                    for i in range(nu):
                        #use curr_u vector from above
                        #TODO: confirm that indices are consistent
                        h[i+4] = curr_u[i]
                    for i in range(nu, 2*nu):
                        #use curr_u vector from above
                        #TODO: confirm that indices are consistent
                        h[i+4] = 1 - curr_u[i-nu]

                    print(P)
                    P = cvxopt_matrix(P, tc='d')
                    q = cvxopt_matrix(q, tc='d')
                    A = cvxopt_matrix(A, tc='d')
                    b = cvxopt_matrix(b, tc='d')
                    G = cvxopt_matrix(G, tc='d')
                    h= cvxopt_matrix(h, tc='d')

                    #soln = cvxopt_solvers.qp(P, q, G, h, A, b)
                    print("normal case")
                    print("P", P)
                    print("q", q)
                    print("G", G)
                    print("h", h)
                    print("u", u)
                    print("p", p)
                    print("u-p", u-p)
                    #breakpoint()
                    soln = cvxopt_solvers.qp(P, q, G, h, options = opts)
                    sols = np.array(soln['x']).reshape(1,-1)[0]
                    print("SOLUTION: ", sols)
                    #breakpoint()
                    solns = sols

                k = -np.matmul(np.atleast_1d(inv_q_uu), np.atleast_1d(q_u))
                knew = np.atleast_1d(solns)
                print("k is: ", k)
                print("knew is: ", knew)
                k = knew
            elif self.constraint_type == 'None':
                k = -np.matmul(np.atleast_1d(inv_q_uu), np.atleast_1d(q_u))
            else:
                raise(NotImplementedError)
                
            kk = -np.matmul(np.atleast_1d(inv_q_uu), np.atleast_1d(q_ux))
            dv = 0.5 * np.matmul(np.atleast_1d(q_u), np.atleast_1d(k))
            self.v[l] += dv
            self.v_x[l] = q_x - np.matmul(np.matmul(np.atleast_1d(q_u), np.atleast_1d(inv_q_uu)), np.atleast_1d(q_ux))
            self.v_xx[l] = q_xx + np.matmul(np.atleast_1d(q_ux).T, np.atleast_1d(kk))
            k_seq.append(k)
            kk_seq.append(kk)
            #breakpoint()
        k_seq.reverse()
        kk_seq.reverse()
        print('k_seq: ',k_seq)
        print('kk_seq: ', kk_seq)
        print('v_seq: ',self.v)
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        incoming_u_seq = self.u_seq_to_incoming_u_seq(u_seq_hat)
        incoming_u_seq_hat = np.array(incoming_u_seq)
        alpha=0.1
        incoming_nodes = self.get_incoming_node_list()

        for t in range(self.pred_time):

            incoming_x_seq = self.x_seq_to_incoming_x_seq(x_seq_hat)
            incoming_rewards_arr = incoming_x_seq[t]
            if t in incoming_nodes[t]:
                l_ind = incoming_nodes[t].index(t)
                x = incoming_rewards_arr[l_ind]
                incoming_rewards_arr.pop(l_ind)
                additional_x = incoming_rewards_arr
            else:
                l_ind = -1
                additional_x = incoming_rewards_arr
                x = None
            #breakpoint()
            control = alpha*k_seq[t] + np.atleast_1d(kk_seq[t]) * (np.atleast_1d(x_seq_hat[t]) - np.atleast_1d(x_seq[t]))
            incoming_u_seq_hat[t] = np.clip(incoming_u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f[t](x, incoming_u_seq_hat[t], additional_x,l_ind) # TODO maybe this should be f[t+1]
            u_seq_hat = self.incoming_u_seq_to_u_seq(incoming_u_seq_hat)
            #breakpoint()
        return x_seq_hat, u_seq_hat

    def x_seq_to_incoming_x_seq(self, x_seq):
        """
        :param x_seq: sequence of node reward values, where index i refers to the i'th nodes reward
        :return: incoming_x_seq: a sequence of *incoming* reward values to each node, where the i'th index contains a
                list of the incoming reward values to the i+1th node
        """
        incoming_x_seq = []
        for k in range(1,len(x_seq)):
            in_nodes = [x_seq[i] for i, x in enumerate(self.adjmat[:,k]) if x==1]
            incoming_x_seq.append(in_nodes)
        #breakpoint()
        return incoming_x_seq

    def u_seq_to_incoming_u_seq(self, u_seq):
        """
        :param u_seq: sequence of flows, where index i corresponds to the flow in edge i
        :return: incoming_u_seq: sequence of incoming flows, where index i corresponds to the list of incoming flows over
                the edges that are incident to node i+1.
        """
        incoming_u_seq = []
        for k in range(1,self.pred_time+1):
            in_node_indices = [i for i, x in enumerate(self.adjmat[:,k]) if x==1]
            u_incoming = []
            for in_node_ind in in_node_indices:
                u_incoming.append(float(u_seq[self.edgelist.index([in_node_ind,k])]))
            incoming_u_seq.append(u_incoming)
        #breakpoint()
        return incoming_u_seq

    def get_incoming_node_list(self):
        """
        :return: a list of lists of incoming neighbor nodes, where the i'th entry corresponds to a list of the
        incoming neighbors of node i+1
        """
        incoming_list = []
        for k in range(1,self.pred_time+1):
            in_node_indices = [i for i, x in enumerate(self.adjmat[:,k]) if x==1]
            incoming_list.append(in_node_indices)
        #breakpoint()
        return incoming_list

    def incoming_u_seq_to_u_seq(self, incoming_u_seq):
        u_seq = np.zeros(len(self.edgelist),)
        incoming_node_list = self.get_incoming_node_list()
        for k in range(self.pred_time):
            #breakpoint()
            node_inds = incoming_node_list[k]
            if len(node_inds)>1:
                for j in range(len(node_inds)):
                    ind = self.edgelist.index([node_inds[j],k+1])
                    u_seq[ind] = incoming_u_seq[k][j]
            else:
                ind = self.edgelist.index([node_inds[0],k+1])
                u_seq[ind] = incoming_u_seq[k][0]

        return u_seq


    def compare_func(self, func):
        for i in range(10):
            print(func(np.array([0],dtype=float),np.array([i],dtype=float)))
"""
env = gym.make('CartPoleContinuous-v0').env
obs = env.reset()
ddp = DDP(lambda x, u: env._state_eq(x, u),  # x(i+1) = f(x(i), u)
          lambda x, u: 0.5 * np.sum(np.square(u)),  # l(x, u)
          lambda x: 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3])),  # lf(x)
          env.max_force,
          env.observation_space.shape[0])
u_seq = [np.zeros(1) for _ in range(ddp.pred_time)]
x_seq = [obs.copy()]
for t in range(ddp.pred_time):
    x_seq.append(env._state_eq(x_seq[-1], u_seq[t]))
cnt = 0
while True:
    env.render(mode="rgb_array")
    #import pyglet
    #pyglet.image.get_buffer_manager().get_color_buffer().save('frame_%04d.png' % cnt)
    for _ in range(3):
        k_seq, kk_seq = ddp.backward(x_seq, u_seq)
        x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)
    print(u_seq.T)
    obs, _, _, _ = env.step(u_seq[0])
    x_seq[0] = obs.copy()
    cnt += 1
"""
