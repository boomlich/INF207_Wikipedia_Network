import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import approximation as approx
import time
import numpy as np
import random


G = nx.DiGraph()

def draw_box_chart(data: list[(str, int)], filepath: str, x_label, y_label):
    names, count = zip(*data)

    plt.figure(figsize=(10, 6))
    plt.bar(names, count, color='skyblue')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()

    plt.savefig(filepath, format=filepath.split('.')[-1], dpi=300)


def generate_graph_from_file(filename: str):
    n = 0

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            nodes = line.strip().split('$$')

            main_node = nodes[0]
            neighbors = nodes[1:]

            for neighbor in neighbors:
                G.add_edge(main_node, neighbor)

            n += 1

            if n % 10000 == 0:
                print('.', end='', flush=True)


def diameter(graph):
    start_time = time.time()
    estimated_diameter = approx.diameter(graph)
    end_time = time.time()
    print(f"Estimated Diameter: {estimated_diameter}")
    print(f"Elapsed Time: {end_time - start_time} seconds")


def shortest_distances(graph, source_node):
    lengths = nx.single_source_shortest_path_length(graph, source_node)
    total_length = sum(lengths.values())
    count = len(lengths) - 1
    return total_length, count


def estimate_avg_shortest_path(graph, sample_size: int = 100) -> float:
    nodes = list(graph.nodes())
    if sample_size > len(nodes):
        sample_size = len(nodes)

    sampled_nodes = random.sample(nodes, sample_size)

    print()
    print(f'Calculating shortest distances with sample size of {sample_size}')
    results = []
    start_time = time.time()
    for i, sample_node in enumerate(sampled_nodes):
        results.append(shortest_distances(graph, sample_node))
        if i > 0 and i % 10 == 0:
            print('.', end='', flush=True)
        if i > 0 and i % 100 == 0:
            print('|', end='', flush=True)
    end_time = time.time()
    print()
    print(f"Elapsed Time: {end_time - start_time} seconds")

    total_path_length = sum([length for length, _ in results])
    path_count = sum([count for _, count in results])

    if path_count == 0:
        return float('inf')
    else:
        average_length = total_path_length / path_count
        return average_length


def get_all_with_dist_from_node(graph, start_node: str, end_node: str):

    return nx.shortest_path(graph, source=start_node, target=end_node)

def graph_analysis():
    scc = nx.strongly_connected_components(G)
    scc_sorted = sorted(scc, key=len, reverse=True)

    largest_scc = scc_sorted[0]

    subgraph = G.subgraph(largest_scc)

    out_degrees = dict(subgraph.out_degree())
    average_out_degree = sum(out_degrees.values()) / subgraph.number_of_nodes()

    print()
    print(f'Number of edges: {len(G.edges())}')
    print(f'Number of nodes: {len(G.nodes())}')
    print(f"Average out-degree in the largest SCC: {average_out_degree}")
    print(f"Number of scc: {len(scc_sorted)}")

    start_node = 'Universitetet i Bergen'
    end_node = 'Ringenes herre'
    uib_lort_path = get_all_with_dist_from_node(subgraph, start_node, end_node)

    print()
    print(f"Shortest path from '{start_node}' to '{end_node}: {len(uib_lort_path)}")
    print("With path:")
    print(uib_lort_path)
    print()

    diameter(subgraph)
    sample_size = 1000
    estimated_avg_dist = estimate_avg_shortest_path(subgraph, sample_size)

    output = 'Size of the 5 largest SCC in graph\n'
    output += '\n'.join([f'{i}: {len(component)}' for i, component in enumerate(scc_sorted[:5], 1)])
    output += '\n'
    output += f'Estimated avg. smallest path ({sample_size}): {estimated_avg_dist}\n'

    with open('graph_analysis.txt', 'w') as f:
        f.write(output)

    print(f'Estimated avg smallest distance in largest SCC: {estimated_avg_dist}')


def degree_distribution():
    degrees = [degree for node, degree in G.degree()]
    degree_counts = np.bincount(degrees)
    degree_values = np.nonzero(degree_counts)[0]

    plt.figure()
    plt.loglog(degree_values, degree_counts[degree_values], 'bo')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()


def largest_degrees():
    in_degrees = G.in_degree()
    out_degrees = G.out_degree()

    sorted_in_degrees = sorted(in_degrees, key=lambda x: x[1], reverse=True)
    sorted_out_degrees = sorted(out_degrees, key=lambda x: x[1], reverse=True)

    print("#" * 20)
    print("Top 10 nodes with highest in-degree:")
    draw_box_chart(
        sorted_in_degrees[:10],
        'top_in_degrees.jpg',
        x_label='Artikkel',
        y_label='Antall inn-grader'
    )

    for node, degree in sorted_in_degrees[:10]:
        print(f"Node {node}: In-Degree {degree}")

    print("#" * 20)
    print("\nTop 10 nodes with highest out-degree:")
    for node, degree in sorted_out_degrees[:10]:
        print(f"Node {node}: Out-Degree {degree}")


def draw_universities():
    universities = {
        'Universitetet i Oslo': 'UiO',
        'Universitetet i Bergen': 'UiB',
        'Norges teknisk-naturvitenskapelige universitet': 'NTNU',
        'Norges handelshøyskole': 'NHH',
        'Handelshøyskolen BI': 'BI',
        'Oslomet – storbyuniversitetet': 'Oslomet',
        'Universitetet i Stavanger': 'UiS',
        'Høgskulen på Vestlandet': 'HVL'
    }

    unis = [(v, G.in_degree(k)) for k, v in universities.items()]
    unis.sort(key=lambda x: x[1], reverse=True)

    draw_box_chart(
        unis,
        'uni.jpg',
        'Universiteter',
        'Antall grader',
    )


def years():
    stats = [(year, G.in_degree(str(year)), G.out_degree(str(year))) for year in range(1800, 2030)]
    years = [year for year, in_deg, out_deg in stats]
    in_degrees = [in_deg for _, in_deg, _ in stats]

    plt.figure(figsize=(12, 6))
    plt.plot(years, in_degrees, label='In Degree', color='blue', linewidth=1.5)
    plt.title('In-Degree and Out-Degree Over Years (1900-2023)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Degree', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

def find_union(a: str, b: str):
    union_ab = []
    b_list = [x for x in G.predecessors(b)]

    for x in G.predecessors(a):
        if x in b_list:
            union_ab.append(x)

    return union_ab


if __name__ == '__main__':
    generate_graph_from_file("wikipedia_graph.txt")
    graph_analysis()
    degree_distribution()
    largest_degrees()
    years()
    draw_universities()

    a = "1802"
    b = "Chrysomeloidea"
    union = find_union('1802', "Chrysomeloidea")

    print(f"Size of union of {a} and {b}: {len(union)}")
