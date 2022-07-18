import pymunk
import pymunk.pygame_util
import pygame
import random as rand
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import toml
import sys

def choose_num_layers(lb=2,ub=9):
    return rand.randint(lb, ub+1)

def choose_layer_height(mean=60):
    h = np.random.normal(loc=mean, scale=mean/2)
    if(h < 30):
        h = 30
    if(h > (2*mean) - 30):
        h = (2*mean) - 30
    return int(h)

def choose_layer_width(ub, num_layers):
    return rand.randint(int(ub - (ub/num_layers)), int(ub))

def choose_block_widths(layer_width, lb=30, ub=200):
    widths = []
    width_left = layer_width
    while(width_left > ub + lb):
        new_width = rand.randint(lb, ub)
        widths.append(new_width)
        width_left -= new_width
    widths.append(width_left)
    return widths

def block_widths_to_ranges(block_widths):
    block_ranges = []
    for layer_blocks in block_widths:
        currpos = 0
        layer_ranges = []
        for width in layer_blocks:
            layer_ranges.append([currpos, currpos + width, width])
            currpos += width
        block_ranges.append(layer_ranges)
    return block_ranges

def generate_tower(initial_width, num_layers):
    initial_height = choose_layer_height()
    initial_block_widths = choose_block_widths(initial_width)
    layer_widths = [initial_width]
    layer_heights = [initial_height]
    layer_blocks = [initial_block_widths]
    for i in range(num_layers-1):
        layer_heights.append(choose_layer_height())
        layer_widths.append(choose_layer_width(layer_widths[-1], num_layers-1))
        layer_blocks.append(choose_block_widths(layer_widths[-1]))
    return layer_widths, layer_heights, block_widths_to_ranges(layer_blocks)

def generate_graph_from_block_info(block_info):
    #list of tuples: (supporting block, supported block)
    edges = []
    contact_edges = []
    interlayer_edges = []
    nodes = set()
    for i in range(len(block_info)-1):
        bottomlayer = block_info[i]
        toplayer = block_info[i+1]
        for j in range(len(bottomlayer)):
            bot_nodename = str(i+1)+","+str(j+1)
            nodes.add(bot_nodename)
            b_start = bottomlayer[j][0]
            b_end = bottomlayer[j][1]
            for k in range(len(toplayer)):
                t_start = toplayer[k][0]
                t_end = toplayer[k][1]
                top_nodename = str(i+2)+","+str(k+1)
                if (t_start >= b_start and t_end <= b_end) or (t_start <= b_start and t_end >= b_end) or (t_start >= b_start and t_start < b_end) or (t_end <= b_end and t_end > b_start):
                    edges.append((bot_nodename, top_nodename))
                    contact_edges.append((bot_nodename, top_nodename))
                else:
                    edges.append((bot_nodename, top_nodename))
                    interlayer_edges.append((bot_nodename, top_nodename))
    
    #add start and end nodes
    nodes.add("S")
    nodes.add("E")

    #adding edges from Start to first layer nodes
    for i in range(len(block_info[0])):
        nodename = str(1)+","+str(i+1)
        edges.append(("S", nodename))

    #adding nodes for last layer and edges to End node
    for i in range(len(block_info[-1])):
        l = len(block_info)
        nodename = str(l)+","+str(i+1)
        nodes.add(nodename)
        edges.append((nodename, "E"))
        

    return list(nodes), edges, contact_edges, interlayer_edges, len(block_info)

def plot_tower_graph(Gr, num_layers, tower_edges, tower_contact_edges, tower_interlayer_edges, save, filename):
    myposdict = {}
    for node in Gr.nodes():
        if node == "S":
            myposdict[node] = (0,0)
        elif node == "E":
            myposdict[node] = (0, num_layers + 1)
        else:
            myposdict[node] = (int(node.split(",")[1])-1, int(node.split(",")[0]))

    pos = myposdict
    nx.draw_networkx_nodes(Gr, pos)
    nx.draw_networkx_labels(Gr, pos)
    nx.draw_networkx_edges(Gr, pos, edgelist=tower_edges, arrowstyle="->", edge_color="black", arrowsize=10)
    nx.draw_networkx_edges(Gr, pos, edgelist=tower_interlayer_edges, arrowstyle="->", edge_color="lightgray", arrowsize=10)
    nx.draw_networkx_edges(Gr, pos, edgelist=tower_contact_edges, arrowstyle="->", edge_color="red", arrowsize=10)

    if save:
        plt.savefig("./autonomous_construction/generated_examples/"+filename+"_graph.png")
    else:
        plt.show()

def draw_tower(layer_heights, block_info, mass_arr, draw):
    GRAY = (50, 50, 50)

    if draw:
        pygame.init()
        size = 700, 900
        screen = pygame.display.set_mode(size)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space()
    #space.gravity = 0, 900

    b0 = space.static_body

    base = pymunk.Segment(b0, (0, 904), (700, 904), 4)
    base.elasticity = 0

    left = pymunk.Segment(b0, (-5, 0), (-5, 900), 4)
    left.elasticity = 0

    right = pymunk.Segment(b0, (704, 0), (704, 900), 4)
    right.elasticity = 0

    space.add(base, left, right)
    
    for i in range(len(block_info)):
        layer_blocks = block_info[i]
        curr_height = layer_heights[i]
        past_heights = sum(layer_heights[:i])
        y_center = 900 - (past_heights + curr_height/2)
        p = 0
        layer_masses = []
        for block in layer_blocks:
            curr_width = block[2]
            x_center = block[0] + (curr_width/2)
            body = pymunk.Body()
            body.position = x_center, y_center
            box = pymunk.Poly.create_box(body, (curr_width, curr_height))
            box.density = 1
            box.elasticity = 0
            if (p + i) % 2 == 0:
                box.color = (0, (255*(i+1))/len(block_info), (255*(i+1))/len(block_info), 255)
            else:
                box.color = (0, 0, (255*(i+1))/len(block_info), 255)
            p += 1
            space.add(body, box)
            layer_masses.append(box.mass)
        mass_arr.append(layer_masses)

    if draw:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(GRAY)
            space.debug_draw(draw_options)
            pygame.display.update()
            #change the step value to get things moving.
            space.step(0.0001)

        pygame.quit()


def export_toml_files(G, draw, heights, blocks, tower_contact_edges, tower_interlayer_edges, filename):
    masses = []
    draw_tower(heights, blocks, masses, draw)
    print(masses)
    #maxmass is the sum over any one layer
    maxmass = max(map(sum, masses))
    for i in range(len(masses)):
        for j in range(len(masses[i])):
            masses[i][j] = masses[i][j]/(maxmass)

    masses = [m for sub in masses for m in sub]
    print(masses)

    nodelist = sorted(list(G.nodes()))
    nodelist.remove("S")
    nodelist.remove("E")
    nodelist.insert(0, "S")
    nodelist.append("E")
    edgelist = sorted([[nodelist.index(e[0]), nodelist.index(e[1])] for e in G.edges()])
    contact_edgelist = sorted([[nodelist.index(e[0]), nodelist.index(e[1])] for e in tower_contact_edges])
    interlayer_edgelist = sorted([[nodelist.index(e[0]), nodelist.index(e[1])] for e in tower_interlayer_edges])
    coalition_types = ['null'] + (len(nodelist)-2)*['sigmoid'] + ['polynomial']
    coalition_params = [[0,0,0]]
    for m in masses:
        coalition_params.append([100*m,10.0,m])
    coalition_params.append([0.0,1.0,0.0])

    dependency_types = len(edgelist)*['polynomial']

    dependency_params = []
    for e in edgelist:
        if e in contact_edgelist:
            dependency_params.append([0.0, 0.0, 0.0])
        elif e in interlayer_edgelist:
            dependency_params.append([0.0, 0.0, 0.0])
        else:
            dependency_params.append([0.0, 0.0, 0.0])

    aggs = len(nodelist)*["or"]

    toml_dict = {
        'exp': {
            'max_steps': 100,
            'edges': edgelist,
            'num_tasks': len(nodelist),
            'numrobots': 4,
            'coalition_types': coalition_types,
            'coalition_params': coalition_params,
            'dependency_types': dependency_types,
            'dependency_params': dependency_params,
            'aggs': aggs
        },
        'ddp': {
            'constraint_type': 'qp'
        },
        'tower': {
            'heights': heights,
            'blocks': blocks
        }
    }

    f = open("./autonomous_construction/generated_examples/"+filename+".toml", 'w+')
    toml.dump(toml_dict, f)

def generate_full_example_data(initial_width, num_layers, filename):
    widths, heights, blocks = generate_tower(initial_width, num_layers)
    tower_nodes, tower_edges, tower_contact_edges, tower_interlayer_edges, n_layers = generate_graph_from_block_info(blocks)
    G = nx.DiGraph()
    G.add_nodes_from(tower_nodes)
    G.add_edges_from(tower_edges)
    plot_tower_graph(G, n_layers, tower_edges, tower_contact_edges, tower_interlayer_edges, True, filename)
    export_toml_files(G, True, heights, blocks, tower_contact_edges, tower_interlayer_edges, filename)

def main():
    if(len(sys.argv) != 4):
        print("Should be called as python initial_blocks.py [initial base width] [number of layers] [filename]")
    generate_full_example_data(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

if __name__ == '__main__':
    main()



""" #print(box.mass, box.moment, box.center_of_gravity)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(GRAY)
    space.debug_draw(draw_options)
    pygame.display.update()
    #change the step value to get things moving.
    space.step(0.01)

pygame.quit() """