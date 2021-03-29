#!/usr/bin/env python
import numpy as np
import warnings
# # monkey patch warnings
# import colorama
# def custom_formatwarning(msg, *args, **kwargs):
#     # ignore everything except the message
#     return colorama.Fore.RED + "Warning: " + str(msg) + '\n' + colorama.Style.RESET_ALL
# warnings.formatwarning = custom_formatwarning

from scipy.interpolate import BSpline


def _save_cast_float_to_int(num):
    if isinstance(num, float) and np.isclose(num, int(num)):
        return int(num)
    else:
        return num


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_unique_nodes(edge_list):
    """
    Using numpy.unique promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.
    """
    return list(set(_flatten(edge_list)))


def _edge_list_to_adjacency_matrix(edge_list, edge_weights=None, unique_nodes=None):

    sources = [s for (s, _) in edge_list]
    targets = [t for (_, t) in edge_list]
    if edge_weights:
        weights = [edge_weights[edge] for edge in edge_list]
    else:
        weights = np.ones((len(edge_list)))

    if unique_nodes is None:
        # map nodes to consecutive integers
        nodes = sources + targets
        unique_nodes = set(nodes)

    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))

    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


def _edge_list_to_adjacency_list(edge_list):
    adjacency = dict()
    for source, target in edge_list:
        if source in adjacency:
            adjacency[source] |= set([target])
        else:
            adjacency[source] = set([target])
    return adjacency


def _get_subgraph(edge_list, node_list):
    subgraph_edge_list = [(source, target) for source, target in edge_list if (source in node_list) and (target in node_list)]
    return subgraph_edge_list


# Adapted from https://stackoverflow.com/a/35007804/2912349
def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """

    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))


def get_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = get_unit_vector(v1)
    v2_u = get_unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / (2 * np.pi) * 360


def get_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def _get_text_object_dimensions(ax, string, *args, **kwargs):
    text_object = ax.text(0., 0., string, *args, **kwargs)
    renderer = _find_renderer(text_object.get_figure())
    bbox_in_display_coordinates = text_object.get_window_extent(renderer)
    bbox_in_data_coordinates = bbox_in_display_coordinates.transformed(ax.transData.inverted())
    w, h = bbox_in_data_coordinates.width, bbox_in_data_coordinates.height
    text_object.remove()
    return w, h


def _find_renderer(fig):
    """
    https://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend
    """

    if hasattr(fig.canvas, "get_renderer"):
        # Some backends, such as TkAgg, have the get_renderer method, which
        # makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        # Other backends do not have the get_renderer method, so we have a work
        # around to find the renderer. Print the figure to a temporary file
        # object, and then grab the renderer that was used.
        # (I stole this trick from the matplotlib backend_bases.py
        # print_figure() method.)
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return(renderer)
