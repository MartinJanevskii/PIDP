import numpy as np
import heapq
import networkx as netx
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def load_map(filepath):
    return np.loadtxt(filepath, delimiter=',')

def plot_map(grid, path=None):
    plt.imshow(grid, cmap='Reds', origin='upper')
    if path:
        for (x, y) in path:
            plt.plot(y, x, 'go')
    plt.show()


def split_map(grid, tile_size):
    """Split the map into smaller regions."""
    height, width = grid.shape
    tiles = []
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            region = grid[i:i + tile_size, j:j + tile_size]
            tiles.append(((i, j), region))
    return tiles

def create_region_graph(region, offset, grid):
    """Convert a region into a graph and validate boundary nodes."""
    graph = netx.Graph()
    rows, cols = region.shape
    boundary_nodes = set()  # To store validated boundary nodes

    for x in range(rows):
        for y in range(cols):
            global_x, global_y = x + offset[0], y + offset[1]
            if region[x, y] == 0:  # Traversable
                graph.add_node((global_x, global_y))

                # Check neighbors and add edges
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and region[nx, ny] == 0:
                        graph.add_edge((global_x, global_y), (nx + offset[0], ny + offset[1]))
                    elif nx < 0 or ny < 0 or nx >= rows or ny >= cols:
                        # Neighbor outside region: Check if it connects to a 0 in the neighboring region
                        neighbor_x, neighbor_y = global_x + dx, global_y + dy
                        if 0 <= neighbor_x < grid.shape[0] and 0 <= neighbor_y < grid.shape[1]:
                            if grid[neighbor_x, neighbor_y] == 0:
                                boundary_nodes.add((global_x, global_y))
    return graph, boundary_nodes




def a_star_search(graph, start, goal):
    open_set = [(0, start)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_set:
        r, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (g_score[neighbor], neighbor))
    return None

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def parallel_a_star(region_graphs, start_goal_pairs):
    print(f"Start/Goal Pairs: {start_goal_pairs}")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(a_star_search, graph, start, goal)
            for graph, (start, goal) in zip(region_graphs, start_goal_pairs)
        ]
        results = [f.result() for f in futures]
        for i, result in enumerate(results):
            print(f"Region {i} Path: {result}")
        return results

def link_regions_and_merge(region_paths, all_boundary_nodes):
    """Link paths across regions using boundary nodes."""
    global_path = []
    visited_regions = set()

    for i, path in enumerate(region_paths):
        if path:
            global_path.extend(path)  # Avoid duplicates at boundaries

            # Debugging boundary node inclusion
            print(f"Processing region {i}, Path: {path}")
            for node in path:
                for offset, boundary_nodes in all_boundary_nodes.items():
                    if node in boundary_nodes and offset not in visited_regions:
                        print(f"Crossing boundary at node: {node} to region offset: {offset}")
                        visited_regions.add(offset)
                        break

    # Ensure the final goal is added if the last region has a path
    if region_paths[-1]:
        global_path.append(region_paths[-1][-1])  # Add the goal from the last region
    else:
        print("Warning: Final region path is None. Check region boundaries or connectivity.")
    return global_path




def main():
    grid = load_map('testing/test1.csv')
    plot_map(grid)

    tile_size = 3
    regions = split_map(grid, tile_size)

    region_graphs = []
    all_boundary_nodes = {}

    for (offset, region) in regions:
        graph, boundary_nodes = create_region_graph(region, offset, grid)
        region_graphs.append(graph)
        all_boundary_nodes[offset] = boundary_nodes

    # Define global start and goal
    start = (0, 0)
    goal = (5, 5)

    # Map start/goal to the correct regions
    start_goal_pairs = []
    for (offset, region) in regions:
        # Check if the start or goal belongs to this region
        start_in_region = offset[0] <= start[0] < offset[0] + region.shape[0] and \
                          offset[1] <= start[1] < offset[1] + region.shape[1]
        goal_in_region = offset[0] <= goal[0] < offset[0] + region.shape[0] and \
                         offset[1] <= goal[1] < offset[1] + region.shape[1]

        # Determine regional start/goal
        regional_start = start if start_in_region else None
        regional_goal = goal if goal_in_region else None

        # Assign start/goal based on region type
        if start_in_region:
            # Start Region: Use boundary nodes as intermediate goals
            boundary_node = next(iter(all_boundary_nodes[offset]), None)
            start_goal_pairs.append((regional_start, boundary_node))
        elif goal_in_region:
            # Goal Region: Use boundary nodes as intermediate starts
            boundary_node = next(iter(all_boundary_nodes[offset]), None)
            start_goal_pairs.append((boundary_node, regional_goal))
        else:
            # Intermediate Region: Connect boundary nodes
            boundary_nodes = list(all_boundary_nodes[offset])
            if len(boundary_nodes) >= 2:
                start_goal_pairs.append((boundary_nodes[0], boundary_nodes[1]))
            else:
                start_goal_pairs.append((None, None))  # No valid path in this region

    print("Start/Goal Pairs:", start_goal_pairs)

    # Run A* for each region in parallel
    region_paths = parallel_a_star(region_graphs, start_goal_pairs)

    # Link boundary nodes and merge paths
    combined_path = link_regions_and_merge(region_paths, all_boundary_nodes)
    print("Global Path:", combined_path)
    plot_map(grid, combined_path)


if __name__ == "__main__":
    main()
