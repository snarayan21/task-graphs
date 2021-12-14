#import gym
#import env
from autograd import grad, jacobian
import autograd.numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class DDP:
    def __init__(self, next_state, #vector of dynamics equations function handles -- one for each node
                 running_cost,  #vector of running cost function handles -- one for each node
                 final_cost, #function of final cost
                 umax, 
                 state_dim, 
                 pred_time,
                 inc_mat,
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
        self.constraint_type = constraint_type


    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for l in range(self.pred_time - 1, -1, -1):
            l_x = grad(self.l, 0)
            l_u = grad(self.l, 1)
            l_xx = jacobian(l_x, 0)
            l_uu = jacobian(l_u, 1)
            l_ux = jacobian(l_u, 0)
            f_x = jacobian(self.f[l], 0)
            f_u = jacobian(self.f[l], 1)
            f_xx = jacobian(f_x, 0)
            f_uu = jacobian(f_u, 1)
            f_ux = jacobian(f_u, 0)
            f_x_t = f_x(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l]))
            f_u_t = f_u(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l]))
            q_x = l_x(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])) + np.matmul(np.atleast_1d(f_x_t.T), np.atleast_1d(self.v_x[l + 1]))
            q_u = l_u(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])) + np.matmul(np.atleast_1d(f_u_t.T), np.atleast_1d(self.v_x[l + 1]))
            q_xx = l_xx(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])) + \
              np.matmul(np.atleast_1d(np.matmul(np.atleast_1d(f_x_t.T), np.atleast_1d(self.v_xx[l + 1]))), np.atleast_1d(f_x_t)) + \
              np.dot(np.atleast_1d(self.v_x[l + 1]), np.atleast_1d(np.squeeze(f_xx(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])))))
            tmp = np.matmul(np.atleast_1d(f_u_t.T), np.atleast_1d(self.v_xx[l + 1]))
            q_uu = l_uu(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])) + np.matmul(np.atleast_1d(tmp), np.atleast_1d(f_u_t)) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_uu(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l]))))
            q_ux = l_ux(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l])) + np.matmul(np.atleast_1d(tmp), np.atleast_1d(f_x_t)) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_ux(np.atleast_1d(x_seq[l]), np.atleast_1d(u_seq[l]))))

            try:
                inv_q_uu = np.linalg.inv(np.atleast_2d(q_uu))
            except np.linalg.LinAlgError:
                inv_q_uu = np.array([[0.0]])
                print('SINGULAR MATRIX: RETURNING ZERO GRADIENT')
            print('Quu: ', q_uu)
            q_uu = np.atleast_2d(q_uu)
            q_x = np.atleast_2d(q_x)
            
            if self.constraint_type == 'qp':
                nu, _ = q_uu.shape
                print(q_x)
                curr_inc_mat = self.incmat[l+1]
                print(curr_inc_mat)
                #curr node inflow
                u = 0.0
                for i in range(len(curr_inc_mat)):
                    if(curr_inc_mat[i] == 1):
                        u += u_seq[i]
                #curr node outflow
                p = 0.0
                for i in range(len(curr_inc_mat)):
                    if(curr_inc_mat[i] == -1):
                        p += u_seq[i]

                solns = np.zeros(nu)

                if(l == self.pred_time - 1):
                    #constraint on p doesn't take place if we are at last node. Only z slack variable
                    P = np.copy(q_uu)
                    P = np.hstack((P, np.zeros((P.shape[0], 1))))
                    P = np.vstack((P, np.zeros((1, P.shape[1]))))
                    P = cvxopt_matrix(P, tc='d')
                    q = np.copy(q_x)
                    q = np.vstack((q, np.zeros((1,1))))
                    A = np.ones((1, nu+1))
                    b = np.array([1 - u])
                    G = np.zeros((1, nu+1))
                    G[0][-1] = -1
                    h = np.zeros((1, 1))

                    P = cvxopt_matrix(P, tc='d')
                    q = cvxopt_matrix(q, tc='d')
                    A = cvxopt_matrix(A, tc='d')
                    b = cvxopt_matrix(b, tc='d')
                    G = cvxopt_matrix(G, tc='d')
                    h= cvxopt_matrix(h, tc='d')

                    soln = cvxopt_solvers.qp(P, q, G, h, A, b)
                    sols = np.array(soln['x']).reshape(1,-1)[0]
                    print("SOLUTION: ", sols)
                    solns = sols[:-1]

                else:
                    P = np.copy(q_uu)
                    P = np.hstack((P, np.zeros((P.shape[0], 2))))
                    P = np.vstack((P, np.zeros((2, P.shape[1]))))
                    q = np.copy(q_x)
                    q = np.vstack((q, np.zeros((2,1))))
                    A = np.full((1, nu+2), 2)
                    A[0][-1] = 1
                    A[0][-2] = 1
                    b = np.array([1 + p - (2*u)])
                    G = np.zeros((2, nu+2))
                    G[0][-1] = 1
                    G[1][-1] = -1
                    h = np.zeros((2, 1))

                    P = cvxopt_matrix(P, tc='d')
                    q = cvxopt_matrix(q, tc='d')
                    A = cvxopt_matrix(A, tc='d')
                    b = cvxopt_matrix(b, tc='d')
                    G = cvxopt_matrix(G, tc='d')
                    h= cvxopt_matrix(h, tc='d')

                    soln = cvxopt_solvers.qp(P, q, G, h, A, b)
                    sols = np.array(soln['x']).reshape(1,-1)[0]
                    print("SOLUTION: ", sols)
                    solns = sols[-2]

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
            self.v_xx[l] = q_xx + np.matmul(np.atleast_1d(q_ux.T), np.atleast_1d(kk))
            k_seq.append(k)
            kk_seq.append(kk)
            #breakpoint()
        k_seq.reverse()
        kk_seq.reverse()
        print('k_seq: ',k_seq)
        print('kk_seq: ', kk_seq)
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(np.atleast_1d(kk_seq[t]), (np.atleast_1d(x_seq_hat[t]) - np.atleast_1d(x_seq[t])))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f[t](x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat

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
