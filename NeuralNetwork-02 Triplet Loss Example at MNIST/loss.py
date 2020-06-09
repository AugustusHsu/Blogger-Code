# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:41:35 2020

@author: jimhs
"""

import tensorflow as tf

def _distance_metric(embedding, squared=False):
    """
    Args:
        x: float32, with shape [m, d], (batch_size, d)
        y: float32, with shape [n, d], (batch_size, d)
    Returns:
        dist: float32, with shape [m, n], (batch_size, batch_size)
    """
    # |x-y|^2 = x^2 - 2xy + y^2
    xy = tf.matmul(embedding, tf.transpose(embedding))
    square_norm = tf.linalg.diag_part(xy)
    xx = tf.expand_dims(square_norm, 0)
    yy = tf.expand_dims(square_norm, 1)
    distances = tf.math.add(xx, yy) - 2.0 * xy
    '''
    (batch_size,1)-(batch_size,batch_size): Equivalent to each column operation
    (batch_size,batch_size)+(1,batch_size): Equivalent to each row operation
    '''

    # Deal with numerical inaccuracies. Set small negatives to zero.
    distances = tf.math.maximum(distances, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        distances = tf.math.sqrt(distances + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)

    # Undo conditionally adding 1e-16.
    distances = tf.math.multiply(distances, tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),)

    num_data = tf.shape(embedding)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(distances) - tf.linalg.diag(tf.ones([num_data]))
    distances = tf.math.multiply(distances, mask_offdiagonals)

    return distances

def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: float32, with shape [n, m], (batch_size, batch_size)
      mask: boolean, with shape [n, m], (batch_size, batch_size)
      dim: int, the dimension which want to compute the minimum.
    Returns:
      masked_minimums: float32, with shape [n, 1], (batch_size, batch_size)
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (tf.math.reduce_min(tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True) + axis_maximums)
    return masked_minimums

def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: float32, with shape [n, m], (batch_size, batch_size)
      mask: boolean, with shape [n, m], (batch_size, batch_size)
      dim: int, the dimension over which to compute the maximum.
    Returns:
      masked_minimums: float32, with shape [n, 1], (batch_size, batch_size)
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (tf.math.reduce_max(tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True) + axis_minimums)
    return masked_maximums

def triplet_batch_hard(labels, embedding, margin, soft):
    '''
    batch hard triplet loss of a batch
    ------------------------------------
    Args:
        labels:     Label Data, shape = (batch_size,1)
        embedding:  embedding vector, shape = (batch_size, vector_size)
        margin:     margin, scalar
        soft::     	use log1p or not, boolean
    Returns:
        triplet_loss: scalar, for one batch
    '''
    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    # Build pairwise squared distance matrix.
    pdist_matrix = _distance_metric(embedding, squared=True)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)
    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)

    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)
    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(tf.ones([batch_size]))
    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)
    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss

def triplet_batch_semihard(labels, embedding, margin):
    '''
    semi-hard batch triplet loss of a batch
    ------------------------------------
    Args:
        labels:     label data, shape = (batch_size,1)
        embedding:  embedding vector, shape = (batch_size, vector_size)
        margin:     margin, scalar
    Returns:
        triplet_loss: scalar, for one batch
    '''
    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    # Build pairwise squared distance matrix.
    pdist_matrix = _distance_metric(embedding, squared=True)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)
    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(tf.tile(adjacency_not, [batch_size, 1]),
                               tf.math.greater(pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])),)
    mask_final = tf.reshape(tf.math.greater(tf.math.reduce_sum(tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True),
                                            0.0,),
                            [batch_size, batch_size],)
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(_masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(_masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(tf.math.reduce_sum(tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)),num_positives,)

    return triplet_loss