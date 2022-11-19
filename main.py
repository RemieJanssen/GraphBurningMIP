import pulp as pl
import networkx as nx
import itertools
solver = pl.getSolver('GLPK_CMD')
problem = pl.LpProblem("myProblem", pl.LpMinimize)

graph = nx.DiGraph()
#graph.add_nodes_from([0])
graph.add_edges_from([
    (0,1),
    (0,2),
    (3,0),
    (4,3),
])

no_of_nodes = graph.number_of_nodes()
upper_bound = no_of_nodes

graph_neighbourhood = graph.neighbors
if isinstance(graph, nx.DiGraph):
    nbh = graph.predecessors
    


# Define variables
vars_node_k = {
    (node,k): pl.LpVariable(f"x-{node}-{k}", cat='Binary') 
    for node, k 
    in itertools.product(graph.nodes, range(1, upper_bound+1))
}

vars_node_step = {
    node: pl.LpVariable(f"y-{node}", lowBound=0, cat="Continuous")
    for node in graph.nodes
}

vars_node_nb = dict()
for node in graph.nodes:
    for nb in graph_neighbourhood(node):
        vars_node_nb[(node, nb)] = pl.LpVariable(
            f"z-{node}-{nb}", 
            cat='Binary'
        )



# Constraints

# y_v >= 1
for node in graph.nodes:
    expression = pl.LpAffineExpression([
        (vars_node_step[node], 1) 
    ])
    constraint = pl.LpConstraint(
#        name="y_v >= 1",
        e=expression, 
        sense=pl.LpConstraintGE,
        rhs=1,
    )
    problem += constraint

# sum_v x_v^1  <= 1
expression = pl.LpAffineExpression([
    (vars_node_k[(node, 1)], 1)
    for node in graph.nodes 
])
constraint = pl.LpConstraint(
#    name="sum_v x_v^1  <= 1",
    e=expression, 
    sense=pl.LpConstraintLE,
    rhs=1,
)
problem += constraint

# sum_v x_v^k  <= sum_v x_v^{k-1}
for k in range(2,upper_bound+1):
    positive = [
        (vars_node_k[(node, k)], 1)
        for node in graph.nodes 
    ]
    negative = [
        (vars_node_k[(node, k-1)], -1)
        for node in graph.nodes 
    ]
    expression = pl.LpAffineExpression(positive+negative)
    constraint = pl.LpConstraint(
#        name="sum_v x_v^k  <= sum_v x_v^{k-1}",
        e=expression, 
        sense=pl.LpConstraintLE,
        rhs=0,
    )
    problem += constraint


# y_v <= k x_v^k  + |V| (1-x_v^k)    for all v \in V and  k>0
# i.e. 
# y_v + (|V|-k) x_v^k <= |V|         for all v \in V and  k>0
for (node, k) in itertools.product(graph.nodes, range(1, upper_bound+1)):
    expression = pl.LpAffineExpression([
        (vars_node_step[node], 1),
        (vars_node_k[(node, k)], upper_bound-k)
    ])
    constraint = pl.LpConstraint(
#        name="y_v <= k x_v^k  + |V| (1-x_v^k)",
        e=expression, 
        sense=pl.LpConstraintLE,
        rhs=upper_bound,
    )
    problem += constraint    
    


# y_v <= y_u - 1  + |V| (1-z_v^u)    for all v \in V and u\in N(v)
# i.e.
# y_v - y_u + |V| z_v^u <= |V| - 1    for all v \in V and u\in N(v)
for node in graph.nodes:
    for nb in graph_neighbourhood(node):
        expression = pl.LpAffineExpression([
            (vars_node_step[node], 1),
            (vars_node_step[nb], -1),
            (vars_node_nb[(node, nb)], upper_bound)
        ])
        constraint = pl.LpConstraint(
#            name="y_v <= y_u - 1  + |V| (1-z_v^u)",
            e=expression, 
            sense=pl.LpConstraintLE,
            rhs=upper_bound - 1,
        )
        problem += constraint    


# \sum_{k} x_v^k + \sum_{u} z_v^u   >= 1 for all v \in V \\
for node in graph.nodes:
    expression = pl.LpAffineExpression([
        (vars_node_k[(node, k)], 1)
        for k in range(1, upper_bound+1)
    ] + [
        (vars_node_nb[(node, nb)], 1)
        for nb in graph_neighbourhood(node)
    ])
    constraint = pl.LpConstraint(
#        name="\sum_{k} x_v^k + \sum_{u} z_v^u",
        e=expression, 
        sense=pl.LpConstraintGE,
        rhs=1,
    )
    problem += constraint    




# Objective

# \sum x_v^k
expression = pl.LpAffineExpression([
    (vars_node_k[(node, k)], 1)
    for (node, k)
    in itertools.product(graph.nodes, range(1, upper_bound+1))    
])
objective = pl.LpConstraint(
    e=expression, 
    sense=pl.LpConstraintLE,
)
problem.objective = objective



# Solve

status = problem.solve()

print(problem)

print(f"status: {pl.LpStatus[status]}")
max_k = pl.value(problem.objective)
print("solution: ")
print(f"burning number: {max_k}")
for (node, k), var in vars_node_k.items():
    if pl.value(var) == 0:
        continue
    print(f"node: {node}, step: {k}, neighbourhood_size: {max_k-k}")  

