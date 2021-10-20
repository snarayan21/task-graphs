#import gym
#import env
from autograd import grad, jacobian
import autograd.numpy as np

class DDP:
    def __init__(self, next_state, #vector of dynamics equations function handles -- one for each node
                 running_cost,  #vector of running cost function handles -- one for each node
                 final_cost, #function of final cost
                 umax, state_dim, pred_time=50):
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


    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for l in range(self.pred_time - 1, -1, -1):
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
            f_x_t = f_x(x_seq[l], u_seq[l])
            f_u_t = f_u(x_seq[l], u_seq[l])
            q_x = l_x(x_seq[l], u_seq[l]) + np.matmul(f_x_t.T, self.v_x[l + 1])
            q_u = l_u(x_seq[l], u_seq[l]) + np.matmul(f_u_t.T, self.v_x[l + 1])
            q_xx = l_xx(x_seq[l], u_seq[l]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[l + 1]), f_x_t) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_xx(x_seq[l], u_seq[l])))
            tmp = np.matmul(f_u_t.T, self.v_xx[l + 1])
            q_uu = l_uu(x_seq[l], u_seq[l]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_uu(x_seq[l], u_seq[l])))
            q_ux = l_ux(x_seq[l], u_seq[l]) + np.matmul(tmp, f_x_t) + \
              np.dot(self.v_x[l + 1], np.squeeze(f_ux(x_seq[l], u_seq[l])))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            self.v[l] += dv
            self.v_x[l] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[l] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat
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
