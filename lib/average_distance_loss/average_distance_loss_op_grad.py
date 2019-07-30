import tensorflow as tf
from tensorflow.python.framework import ops
from .average_distance_loss_op import average_distance_loss_grad

@ops.RegisterGradient("Averagedistance")
def _average_distance_grad(op, grad, _):

  diff = op.outputs[1]
  margin = op.get_attr('margin')

  # compute gradient
  data_grad = average_distance_loss_grad(diff, grad, margin)

  return [data_grad, None, None, None, None]  # List of one Tensor, since we have five input
