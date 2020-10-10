import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) # + 0.0000000000000001
    AD = np.dot(A, Dn)
    # test = np.sum(A[:, 0])
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A



# def get_D(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i]**(-1)
#
#     return Dn
#
# def get_spatial_graph(num_node, self_link, inward, outward):
#     I = edge2mat(self_link, num_node)
#     In = edge2mat(inward, num_node)
#     Out = edge2mat(outward, num_node)
#     Dn = get_D(I + In + Out)
#     I = np.dot(I, Dn)
#     In = np.dot(In, Dn)
#     Out = np.dot(Out, Dn)
#     # test = I + In + Out
#     A = np.stack((I, In, Out))
#     return A

def get_spatial_graph(num_node, self_link, inward, outward):

    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_spatial_all_graph(num_node, self_link, inward, outward):

    Dn = np.ones((num_node, num_node))

    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    neighbor_link = inward + outward + self_link

    #all = normalize_digraph(Dn)
    N = normalize_digraph(Dn - edge2mat(neighbor_link, num_node))
    #A = np.stack((I, In, Out, N, all))
    A = np.stack((I, In, Out, N))
    return A

def get_weighted_spatial_graph(num_node, self_link, inward, outward):

    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node)) * 2
    Out = normalize_digraph(edge2mat(outward, num_node)) * 3
    A = np.stack((I, In, Out))
    return A


def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A