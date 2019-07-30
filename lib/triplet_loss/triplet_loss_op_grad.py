import tensorflow as tf
from tensorflow.python.framework import ops
from .triplet_loss_op import triplet_loss_grad

@ops.RegisterGradient("Triplet")
def _triplet_grad(op, grad, _):

  diff = op.outputs[1]
  margin = op.get_attr('margin')

  # compute gradient
  data_grad = triplet_loss_grad(diff, grad, margin)

  return [data_grad, None, None]  # List of one Tensor, since we have three input
