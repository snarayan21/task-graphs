import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import math

def start_finish_to_event_list(s, f):
    n = len(s)
    s = [(i-1, s[i], "s") for i in range(1,n-1)]
    f = [(i-1, f[i], "f") for i in range(1,n-1)]
    events = s + f
    events = sorted(events, key = lambda x: (x[1], x[0]))
    return events

def robots_to_fraction(totrobots, taskrobots):
    taskrobots = [t/float(totrobots) for t in taskrobots]
    return taskrobots

def sigmoid(flow, param):
    return param[0] / (1 + math.e ** (-1 * param[1] * (flow - param[2])))

def graph_tower(s, f, totrobots, taskrobots, layer_heights, block_info, coalitions, fname):
    fracs = robots_to_fraction(totrobots, taskrobots)
    events = start_finish_to_event_list(s, f)
    print("eventlist", events)
    numtasks = len(coalitions)
    t_height = sum(layer_heights)
    t_width = block_info[0][-1][1]
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([0,int(t_height*1.1)])
    ax.set_ylim([-1*(int(t_width*0.1)),int(t_width*1.1)])
    full_blocks = []
    for j, layer in enumerate(block_info):
        layer_unpacked = [ b + [sum(layer_heights[:j]), layer_heights[j]] for b in layer ]
        full_blocks = full_blocks + layer_unpacked
    def animate(i):
        """ ax.clear()
        edone = 0
        print("--------")
        while(edone <= i):
            event = events[edone]
            xstart, _, width, ystart, height = full_blocks[event[0]]
            if(event[2] == "s"):
                #block start event
                r = patches.Rectangle((xstart, ystart), width, height, linewidth=3, edgecolor="gray", facecolor="none", linestyle="--")
                ax.add_patch(r)
            else:
                #block end event
                task_frac = fracs[i]
                stdev = 0.05*(1-task_frac)
                real_frac = np.random.normal(task_frac, stdev)
                if(real_frac < 0):
                    real_frac = 0
                if(real_frac > 1):
                    real_frac = 1
                rew = sigmoid(real_frac, coalitions[event[0]])
                max_rew = coalitions[event[0]][0]
                print("coalition params are:", coalitions[event[0]])
                print("task_frac is: ", task_frac)
                print("rew is:", rew)
                print("maxrew is: ", max_rew)
                print("reward fraction is: ", rew/max_rew)
                colorhex = "0x{:02x}".format(int((rew/max_rew)*255))[2:]
                print("reward_fraction hex is:", colorhex)
                r = patches.Rectangle((xstart, ystart), width, height, linewidth=3, edgecolor="#"+colorhex+"0000", facecolor="#"+colorhex+"0000")
                ax.add_patch(r)
            edone += 1
        ax.set_xlim([0,int(t_height*1.1)])
        ax.set_ylim([-1*(int(t_width*0.1)),int(t_width*1.1)])

        print("--------") """

        event = events[i]
        xstart, _, width, ystart, height = full_blocks[event[0]]
        if(event[2] == "s"):
            #block start event
            r = patches.Rectangle((xstart, ystart), width, height, linewidth=3, facecolor="gray", alpha=0.5)
            ax.add_patch(r)
        else:
            #block end event
            task_frac = fracs[event[0]+1]
            stdev = 0.05*(1-task_frac)
            real_frac = np.random.normal(task_frac, stdev)
            if(real_frac < 0):
                real_frac = 0
            if(real_frac > 1):
                real_frac = 1
            rew = sigmoid(real_frac, coalitions[event[0]+1])
            max_rew = coalitions[event[0]+1][0]
            colorhex = "0x{:02x}".format(int((rew/max_rew)*255))[2:]
            r = patches.Rectangle((xstart, ystart), width, height, linewidth=3, facecolor="#00"+colorhex+"00")
            ax.add_patch(r)

    anim = FuncAnimation(fig, animate, frames=len(events), interval=1000, repeat = False)
    anim.save("./autonomous_construction/generated_examples/"+fname+".mp4", writer='ffmpeg',fps=1)
    plt.show()

def flows_to_taskrobots(flows, edges, numtasks, numrobots):
    node_flows = [ 0.0 for i in range(numtasks) ]
    for i, flow in enumerate(flows):
        dest_edge = edges[i][1]
        node_flows[dest_edge] = node_flows[dest_edge] + flow
    node_flows = [ int(f*numrobots) for f in node_flows ]
    return node_flows


