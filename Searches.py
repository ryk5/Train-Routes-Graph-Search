import time
from math import pi, acos, sin, cos
from tkinter import *
import tkinter as tk

class HeapPriorityQueue():
    def __init__(self):
        self.queue = ["dummy"]  # we do not use index 0 for easy index calulation
        self.current = 1        # to make this object iterable

    def next(self):            # define what __next__ does
        if self.current >=len(self.queue):
         self.current = 1     # to restart iteration later
         raise StopIteration
    
        out = self.queue[self.current]
        self.current += 1
   
        return out

    def __iter__(self):
        return self

        __next__ = next

    def isEmpty(self):
        return len(self.queue) <= 1    # b/c index 0 is dummy

    def swap(self, a, b):
        self.queue[a], self.queue[b] = self.queue[b], self.queue[a]

   # Add a value to the heap_pq
    def push(self, value):
        self.queue.append(value)
        # write more code here to keep the min-heap property
        self.heapUp(len(self.queue)-1)

   # helper method for push      
    def heapUp(self, k):
        if k <= 1:
            return

        if len(self.queue) % 2 == 1 and k == len(self.queue) - 1: # no sibling
            if sum(self.queue[k//2][1:3]) > sum(self.queue[k][1:3]):
                self.swap(k, k//2)
                self.heapUp(k//2)
            return

        if k % 2 == 0:
            parent, sibling = k//2, k+1
        else:
            parent, sibling = k//2, k-1

        if sum(self.queue[k][1:3]) > sum(self.queue[sibling][1:3]):
            child = sibling
        else:
            child = k

        if sum(self.queue[parent][1:3]) > sum(self.queue[child][1:3]):
            self.swap(child, parent)
            self.heapUp(parent)

               
   # helper method for reheap and pop
    def heapDown(self, k, size):
        left, right = 2*k, 2*k+1

        if left == size and sum(self.queue[k][1:3]) > sum(self.queue[size][1:3]): # One child
            self.swap(k, left)
        
        elif right <= size:
            child = (left if sum(self.queue[left][1:3]) < sum(self.queue[right][1:3]) else right)
 
            if sum(self.queue[k][1:3]) > sum(self.queue[child][1:3]):
                self.swap(k, child)
                self.heapDown(child, size)
      
   # make the queue as a min-heap            
    def reheap(self):
        if self.isEmpty():
            return -1

        for k in range((len(self.queue)-1)//2, 0, -1):
            self.heapUp(k)
   
   # remove the min value (root of the heap)
   # return the removed value            
    def pop(self):
        if self.isEmpty():
            return -1
        self.swap (1, len(self.queue) - 1)
        val = self.queue.pop()
        self.heapDown(1, len(self.queue) - 1)
        return val
      
   # remove a value at the given index (assume index 0 is the root)
   # return the removed value   
    def remove(self, index):

        if self.isEmpty():
            return -1
      
        if len (self.queue) == 2:
            val = self.queue.pop()
            self.queue = []
            return val
      
        self.swap (index + 1, len(self.queue) - 1)
        val = self.queue.pop()
        self.heapDown(index + 1, len(self.queue) - 1)

        return val
    
def calc_edge_cost(y1, x1, y2, x2):
    #
    # y1 = lat1, x1 = long1
    # y2 = lat2, x2 = long2
    # all assumed to be in decimal degrees

    # if (and only if) the input is strings
    # use the following conversions

    y1  = float(y1)
    x1  = float(x1)
    y2  = float(y2)
    x2  = float(x2)

    R    = 3958.76 # miles = 6371 km

    y1 *= pi/180.0
    x1 *= pi/180.0
    y2 *= pi/180.0
    x2 *= pi/180.0

    # approximate great circle distance with law of cosines
    return acos( sin(y1)*sin(y2) + cos(y1)*cos(y2)*cos(x2-x1) ) * R



# NodeLocations, NodeToCity, CityToNode, Neighbors, EdgeCost
# Node: (lat, long) or (y, x), node: city, city: node, node: neighbors, (n1, n2): cost
def make_graph(nodes = "rrNodes.txt", node_city = "rrNodeCity.txt", edges = "rrEdges.txt"):
    nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost = {}, {}, {}, {}, {}
    map = {}    # have screen coordinate for each node location
    
    nodes = open(nodes, "r")
    node_city = open(node_city, "r")
    edges = open(edges, "r")

    for line in node_city.readlines():
        args = line.strip().split(" ", 1)
        nodeToCity[args[0]] = args[1]
        cityToNode[args[1]] = args[0]

    for line in nodes.readlines():
        args = line.strip().split(" ")
        nodeLoc[args[0]] = (float(args[1]), float(args[2]))

    for line in edges.readlines():
        args = line.strip().split(" ")
        cost = calc_edge_cost(nodeLoc[args[0]][0], nodeLoc[args[0]][1], nodeLoc[args[1]][0], nodeLoc[args[1]][1])
        edgeCost[(args[0],args[1])] = cost
        edgeCost[(args[1],args[0])] = cost
        
        if args[0] in neighbors:
            neighbors[args[0]]+=[args[1]]
        else:
            neighbors[args[0]] = [args[1]]

        if args[1] in neighbors:
            neighbors[args[1]]+=[args[0]]
        else:
            neighbors[args[1]] = [args[0]]


    for node in nodeLoc: #checks each
        lat = float(nodeLoc[node][0]) #gets latitude
        long = float(nodeLoc[node][1]) #gets long
        modlat = (lat - 10)/60 #scales to 0-1
        modlong = (long+130)/70 #scales to 0-1
        map[node] = [modlat*800, modlong*1200] #scales to fit 800 1200
    
    return [nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost, map]

# Retuen the direct distance from node1 to node2
# Use calc_edge_cost function.
def dist_heuristic(n1, n2, graph):
    return calc_edge_cost(graph[0][n1][0],graph[0][n1][1],graph[0][n2][0],graph[0][n2][1])
    
# Create a city path. 
# Visit each node in the path. If the node has the city name, add the city name to the path.
# Example: ['Charlotte', 'Hermosillo', 'Mexicali', 'Los Angeles']
def display_path(path, graph):
    print([graph[1][node] for node in path if node in graph[1]])

# Using the explored, make a path by climbing up to "s"
# This method may be used in your BFS and Bi-BFS algorithms.
def generate_path(state, explored, graph):
    path = [state]
    cost = 0

    while explored[state] != "s": #"s" is initial's parent
        cost += graph[4][(state,explored[state])]
        state = explored[state]
        path.append(state)

    return path[::-1], cost

# Draw the final shortest path.
# Use drawLine function.
def draw_final_path(ROOT, canvas, path, graph, col='red'):

    for i in range(len(path[1:])):
        drawLine(canvas, *graph[5][path[i]], *graph[5][path[i+1]], col)

    ROOT.update()
    pass
    
def drawLine(canvas, y1, x1, y2, x2, col):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return canvas.create_line(x1, 800 - y1, x2, 800 - y2, fill=col, width=2)

def is_near_line(x, y, x1, y1, x2, y2, threshold=10):
    # Check if point (x, y) is within a certain threshold distance from line (x1, y1) -> (x2, y2)
    line_len = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    if line_len == 0:  # Avoid division by zero
        return False
    dist_to_line = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_len
    return dist_to_line < threshold

def draw_all_edges(ROOT, canvas, graph):
    ROOT.geometry("1200x800")  # sets geometry
    canvas.pack(fill=tk.BOTH, expand=1)  # sets fill expand
    
    # Create a dictionary to hold drawn routes for interaction
    routes = {}

    for n1, n2 in graph[4]:  # graph[4] keys are edge set
        route_id = f"{n1}-{n2}"
        routes[route_id] = drawLine(canvas, *graph[5][n1], *graph[5][n2], 'white')
    
    # Add dynamic hover events
    def on_hover(event):
        # Check which route the mouse is near
        for route_id, line in routes.items():
            # Get the coordinates of the line using the object ID
            coords = canvas.coords(line)
            if len(coords) != 4:
                continue  # Skip if the line doesn't have 4 coordinates
            x1, y1, x2, y2 = coords
            
            # Calculate distance from mouse to line
            if is_near_line(event.x, event.y, x1, y1, x2, y2, threshold=5):
                # Highlight line
                canvas.itemconfig(line, fill='yellow', width=3)
            else:
                canvas.itemconfig(line, fill='white', width=2)

    canvas.bind("<Motion>", on_hover)
    ROOT.update()

def drawLine(canvas, y1, x1, y2, x2, col):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return canvas.create_line(x1, 800 - y1, x2, 800 - y2, fill=col, width=2)

def is_near_line(x, y, x1, y1, x2, y2, threshold=10):
    # Check if point (x, y) is within a bounding box first
    if not (min(x1, x2) - threshold <= x <= max(x1, x2) + threshold and
            min(y1, y2) - threshold <= y <= max(y1, y2) + threshold):
        return False
    
    # Check if point (x, y) is within a certain threshold distance from line (x1, y1) -> (x2, y2)
    line_len = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    if line_len == 0:  # Avoid division by zero
        return False
    dist_to_line = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_len
    return dist_to_line < threshold

def draw_all_edges(ROOT, canvas, graph):
    ROOT.geometry("1200x800")  # sets geometry
    canvas.pack(fill=tk.BOTH, expand=1)  # sets fill expand
    # Create a dictionary to hold drawn routes for interaction
    routes = {}
    for n1, n2 in graph[4]:  # graph[4] keys are edge set
        route_id = f"{n1}-{n2}"
        routes[route_id] = drawLine(canvas, *graph[5][n1], *graph[5][n2], 'white')
    ROOT.update()

def bfs(start, goal, graph, col):
    ROOT = Tk() #creates new tkinter
    ROOT.title("BFS")
    canvas = Canvas(ROOT, background='black') #sets background
    draw_all_edges(ROOT, canvas, graph)

    counter = 0
    frontier, explored = [], {start: "s"}
    frontier.append(start)
    while frontier:
        s = frontier.pop(0)
        if s == goal: 
            path, cost = generate_path(s, explored, graph)
            draw_final_path(ROOT, canvas, path, graph)
            return path, cost
        for a in graph[3][s]:  #graph[3] is neighbors
            if a not in explored:
                explored[a] = s
                frontier.append(a)
                drawLine(canvas, *graph[5][s], *graph[5][a], col)
        counter += 1
        if counter % 1000 == 0: ROOT.update()
    return None
    return None  # Return None if no path is found
def bi_bfs(start, goal, graph, col):

    ROOT = Tk() #creates new tkinter
    ROOT.title("Bi A Star")
    canvas = Canvas(ROOT, background='black') #sets background
    draw_all_edges(ROOT, canvas, graph)

    Q = [[start],[goal]]
    explored = [{start:"s"},{goal:"s"}]
    flag = 0
    counter = 0

    while Q[0] or Q[1]:
        flag = 1 - flag
        for i in range(len(Q[flag])):
            state = Q[flag].pop(0)
            adj_list = graph[3][state]
            
            for adj in adj_list:
                if adj in explored[1 - flag]:
                    explored[flag][adj] = state
                    start_path, start_cost = generate_path(adj, explored[0], graph)
                    goal_path, goal_cost = generate_path(adj, explored[1], graph)
                    path = start_path[:-1] + goal_path[::-1]

                    draw_final_path(ROOT, canvas, path, graph)
                    return path, start_cost + goal_cost
            
            for neighbor in adj_list:
                if neighbor not in explored[flag]:
                    explored[flag][neighbor] = state
                    Q[flag].append(neighbor)
                    drawLine(canvas, *graph[5][state], *graph[5][neighbor], col)
            
            counter += 1
            if counter % 1000 == 0: ROOT.update()
    return None

def a_star(start, goal, graph, col, heuristic=dist_heuristic):

    ROOT = Tk() #creates new tkinter
    ROOT.title("A*")
    canvas = Canvas(ROOT, background='black') #sets background
    draw_all_edges(ROOT, canvas, graph)

    frontier = HeapPriorityQueue()
    explored = {}
    if start == goal: return []

    # We are pushing tuples of the (current_node, heuristic, path_cost, path))
    frontier.push((start, heuristic(start, goal, graph), 0, [start]))
    explored[start] = heuristic(start, goal, graph)
    
    counter = 0


    while not frontier.isEmpty():
        state = frontier.pop()

        # Goal test
        if state[0] == goal:
            path = state[3]
            draw_final_path(ROOT, canvas, state[3], graph)
            return path, state[2]


        # Push children on heapq
        #print(state[1] + state[2])
        for neighbor in graph[3][state[0]]:
            h = heuristic(neighbor, goal, graph)
            path_cost = graph[4][(neighbor, state[0])] + state[2]
            cost = h + path_cost
            if neighbor not in explored or explored[neighbor] > cost:
                explored[neighbor] = cost
                frontier.push((neighbor, h, path_cost, state[3] + [neighbor]))
                drawLine(canvas, *graph[5][state[0]], *graph[5][neighbor], col)
 
        counter += 1
        if counter % 1000 == 0: ROOT.update()
    
    return None

def bi_a_star(start, goal, graph, col, ROOT, canvas, heuristic=dist_heuristic):

    counter = 0
    frontier = [HeapPriorityQueue(),HeapPriorityQueue()]
    explored = [{start: (heuristic(start, goal, graph), [start])},{goal: (heuristic(goal, start, graph), [goal])}]
    flag = 1

    # We are pushing tuples of the (current_node, heuristic, path_cost, path))
    frontier[0].push((start, heuristic(start, goal, graph), 0, [start]))
    frontier[1].push((goal, heuristic(goal, start, graph), 0, [goal]))

    while frontier[0] or frontier[1]:
        flag = 1 - flag
        state = frontier[flag].pop()

        if state[0] in explored[1 - flag]:
            if flag == 0:
                path = state[3][:-1] + list(reversed(explored[1][state[0]][1]))
            else:
                path = explored[1][state[0]][1] + list(reversed(state[3][:-1]))
            cost = sum([graph[4][(path[i], path[i+1])] for i in range(len(path) - 1)])

            return path, cost
        
        # Push children on heapq
        #print(state[1] + state[2])
        for neighbor in graph[3][state[0]]:
            h = heuristic(neighbor, goal, graph)
            path_cost = graph[4][(neighbor, state[0])] + state[2]
            cost = h + path_cost
            if neighbor not in explored[flag] or explored[flag][neighbor][0] > cost:
                explored[flag][neighbor] = (cost, state[3] + [neighbor])
                frontier[flag].push((neighbor, h, path_cost, state[3] + [neighbor]))
                drawLine(canvas, *graph[5][state[0]], *graph[5][neighbor], col)
        
        counter += 1
        if counter % 1000 == 0: ROOT.update()

    return None

def tri_directional(city1, city2, city3, graph, col, ROOT, canvas, heuristic=dist_heuristic):
    path_1_2, cost_1_2 = bi_a_star(city1, city2, graph, col, ROOT, canvas, heuristic)
    path_1_3, cost_1_3 = bi_a_star(city3, city1, graph, col, ROOT, canvas, heuristic)
    path_2_3, cost_2_3 = bi_a_star(city2, city3, graph, col, ROOT, canvas, heuristic)
    cost, path = min([(cost_1_2+cost_2_3, path_1_2[:-1]+path_2_3), (cost_2_3+cost_1_3, path_2_3[:-1]+path_1_3), (cost_1_3+cost_1_2, path_1_3[:-1]+path_1_2)])
    draw_final_path(ROOT, canvas, path, graph)
    print()
    print(f'Tri-A The whole path: {path}')
    print(f'Tri-A The length of the whole path {len(path)}')
    return path, cost
    
def main():
    start, goal = input("Start city: "), input("Goal city: ")
    third = input("Third city for tri-directional: ")

    graph = make_graph("rrNodes.txt", "rrNodeCity.txt", "rrEdges.txt")  # Task 1
    
    cur_time = time.time()
    path, cost = bfs(graph[2][start], graph[2][goal], graph, 'yellow') #graph[2] is city to node
    if path != None: display_path(path, graph)
    else: print ("No Path Found.")
    print ('BFS Path Cost:', cost)
    print ('BFS duration:', (time.time() - cur_time))
    print ()

    cur_time = time.time()
    path, cost = bi_bfs(graph[2][start], graph[2][goal], graph, 'green')
    if path != None: display_path(path, graph)
    else: print ("No Path Found.")
    print ('Bi-BFS Path Cost:', cost)
    print ('Bi-BFS duration:', (time.time() - cur_time))
    print ()

    cur_time = time.time()
    path, cost = a_star(graph[2][start], graph[2][goal], graph, 'blue')
    if path != None: display_path(path, graph)
    else: print ("No Path Found.")
    print ('A star Path Cost:', cost)
    print ('A star duration:', (time.time() - cur_time))
    print ()
    
    ROOT = Tk() #creates new tkinter
    ROOT.title("Bi A Star")
    canvas = Canvas(ROOT, background='black') #sets background
    draw_all_edges(ROOT, canvas, graph)
    

    cur_time = time.time()
    path, cost = bi_a_star(graph[2][start], graph[2][goal], graph, 'orange', ROOT, canvas)
    if path != None:
        display_path(path, graph)
        draw_final_path(ROOT, canvas, path, graph)
    else: print ("No Path Found.")
    print ('Bi-A star Path Cost:', cost)
    print ("Bi-A star duration: ", (time.time() - cur_time))
    print ()
    
    
    ROOT = Tk() #creates new tkinter
    ROOT.title("Tri A Star")
    canvas = Canvas(ROOT, background='black') #sets background
    draw_all_edges(ROOT, canvas, graph)
    ROOT.mainloop()
    print ("Tri-Search of ({}, {}, {})".format(start, goal, third))
    cur_time = time.time()
    path, cost = tri_directional(graph[2][start], graph[2][goal], graph[2][third], graph, 'pink', ROOT, canvas)
    if path != None: display_path(path, graph)
    else: print ("No Path Found.")
    print ('Tri-A star Path Cost:', cost)
    print ("Tri-directional search duration:", (time.time() - cur_time))

    mainloop() # Let TK windows stay still
 
if __name__ == '__main__':
    main()
