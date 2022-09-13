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
    fig, ax = plt.subplots()
    ax.set_xlim([-1*int(t_width*0.1),int(t_width*1.1)])
    ax.set_ylim([0,int(t_height*1.1)])
    full_blocks = []
    for j, layer in enumerate(block_info):
        layer_unpacked = [ b + [sum(layer_heights[:j]), layer_heights[j]] for b in layer ]
        full_blocks = full_blocks + layer_unpacked
    eventblocks = {}
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
        ax.clear()
        ax.set_xlim([-1*int(t_width*0.1),int(t_width*1.1)])
        ax.set_ylim([0,int(t_height*1.1)])
        for j, event in enumerate(events[:i+1]):
            xstart, _, width, ystart, height = full_blocks[event[0]]
            if(event[2] == "s"):
                #block start event
                if(j not in eventblocks):
                    r = patches.Rectangle((xstart, ystart), width, height, linewidth=3, facecolor="gray", alpha=1)
                    eventblocks[j] = r
                    ax.add_patch(r)
                else:
                    r = eventblocks[j]
                    ax.add_patch(r)

            elif(event[2] == "f"):
                #block end event
                if(j not in eventblocks):
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
                    eventblocks[j] = r
                    ax.add_patch(r)
                else:
                    r = eventblocks[j]
                    ax.add_patch(r)
            
            else:
                pass

    anim = FuncAnimation(fig, animate, frames=int(len(events)), interval=1000, repeat = False)
    anim.save("./autonomous_construction/generated_examples/"+fname+".mp4", writer='ffmpeg',fps=1)
    plt.show()

def graph_tower_image(s, f, layer_heights, block_info, coalitions, fname):


    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n+2)
    
    events = start_finish_to_event_list(s, f)
    print("eventlist", events)
    numtasks = len(coalitions)
    t_height = sum(layer_heights)
    t_width = block_info[0][-1][1]
    fig, axs = plt.subplots(2)
    axs[0].set_xlim([-1*int(t_width*0.1),int(t_width*1.1)])
    axs[0].set_ylim([0,int(t_height*1.1)])
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)
    axs[0].set_title("Block Tower")
    full_blocks = []
    for j, layer in enumerate(block_info):
        layer_unpacked = [ b + [sum(layer_heights[:j]), layer_heights[j]] for b in layer ]
        full_blocks = full_blocks + layer_unpacked
    
    cmap = get_cmap(len(full_blocks))

    for i, block in enumerate(full_blocks):
        xstart, _, width, ystart, height = block
        axs[0].add_patch(patches.Rectangle((xstart, ystart), width, height, facecolor=cmap(i)))

    sftimes = list(zip(s[1:-1], f[1:-1]))

    for i,tasktime in enumerate(sftimes):
        axs[1].barh(i,width=tasktime[1]-tasktime[0],left=tasktime[0], color=cmap(i))

    axs[1].set_yticks(range(len(sftimes)))
    axs[1].set_yticklabels([f'block {i+1}' for i in range(len(sftimes))])
    axs[1].set_xlim(0, max(f))
    axs[1].set_title("Tower Construction Schedule")
    axs[1].set_xlabel("time")

    plt.savefig("./autonomous_construction/"+fname+".png")
    plt.show()

def flows_to_taskrobots(flows, edges, numtasks, numrobots):
    node_flows = [ 0.0 for i in range(numtasks) ]
    for i, flow in enumerate(flows):
        dest_edge = edges[i][1]
        node_flows[dest_edge] = node_flows[dest_edge] + flow
    node_flows = [ int(f*numrobots) for f in node_flows ]
    return node_flows

def graph_tower_image_two(s1, f1, s2, f2, r1, r2, layer_heights, block_info, coalitions, fname):


    def get_cmap(n, name='Set2'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n+2)
    
    numtasks = len(coalitions)
    t_height = sum(layer_heights)
    t_width = block_info[0][-1][1]
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(3, figsize=(5,7))
    axs[0].set_xlim([-1*int(t_width*0.1),int(t_width*1.1)])
    axs[0].set_ylim([0,int(t_height*1.1)])
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)
    axs[0].set_title("Block Tower")
    full_blocks = []
    for j, layer in enumerate(block_info):
        layer_unpacked = [ b + [sum(layer_heights[:j]), layer_heights[j]] for b in layer ]
        full_blocks = full_blocks + layer_unpacked
    
    cmap = get_cmap(len(full_blocks))

    for i, block in enumerate(full_blocks):
        xstart, _, width, ystart, height = block
        axs[0].add_patch(patches.Rectangle((xstart, ystart), width, height, facecolor=cmap(i)))

    sftimes1 = list(zip(s1[1:-1], f1[1:-1]))

    for i,tasktime in enumerate(sftimes1):
        axs[1].barh(i,width=tasktime[1]-tasktime[0],left=tasktime[0], color=cmap(i))

    axs[1].set_yticks(range(len(sftimes1)))
    axs[1].set_yticklabels([f'block {i+1}' for i in range(len(sftimes1))])
    axs[1].set_xlim(0, max(max(f1),max(f2)))
    axs[1].set_title("NLP Solution Schedule (Reward =" + str(r1) + ")")
    axs[1].set_xlabel("time")

    sftimes2 = list(zip(s2[1:-1], f2[1:-1]))

    for i,tasktime in enumerate(sftimes2):
        axs[2].barh(i,width=tasktime[1]-tasktime[0],left=tasktime[0], color=cmap(i))

    axs[2].set_yticks(range(len(sftimes2)))
    axs[2].set_yticklabels([f'block {i+1}' for i in range(len(sftimes2))])
    axs[2].set_xlim(0, max(max(f1),max(f2)))
    axs[2].set_title("MINLP Solution Schedule (Reward =" + str(r2) + ")")
    axs[2].set_xlabel("time")

    plt.subplots_adjust(hspace=0.6)

    plt.savefig("./autonomous_construction/"+fname+".png")
    plt.show()


