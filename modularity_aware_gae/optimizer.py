import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Notes on optimizers:

1 - This code extensively relies on optimizer.py from the tkipf/gae
original repository on standard GAE and VGAE models.

2 - This code proposes two versions of the modularity-inspired loss.
The first one is the standard one, as defined in section 3.3 of the paper.
The second one is a simpler version ignoring the "degree terms", which speeds up results.
This second version is accessible by setting FLAGS.simple as True. While being
mathematically different from the original modularity, we observed few to no
significant changes in LP/CD results when testing it on several of our graphs.
The FastGAE method for scalable GAE/VGAE leverages this simple version by default.

3 - After releasing this paper, we noticed a discrepancy in the code w.r.t. the paper.
While one should normalize the modularity-inspired terms by (1/2*num_edges) as in equations
(18) and (19) of the paper, this code instead normalizes them by (1/num_nodes).
We chose to keep this discrepancy in the source code, to ensure that the optimal beta
hyperparameters correspond to those reported in Table 4 from the paper.
Note: normalizing by (1/2*num_edges) instead would not impact the paper's conclusions.
One would simply have to re-scale beta hyperparameters by a (num_nodes/2*num_edges)
factor to retrieve the optimal models from the paper.
"""


class OptimizerAE(object):

    """
    Optimizer for GAE
    """

    def __init__(self, preds, labels, degree_matrix, num_nodes, pos_weight, norm, clusters_distance):

        preds_sub = preds
        labels_sub = labels

        # Reconstruction term (as in tkipf/gae)
        self.cost_adj =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                                                 labels = labels_sub,
                                                                                 pos_weight = pos_weight))

        # Modularity-inspired term
        if FLAGS.simple or FLAGS.fastgae: # simpler proxy of the modularity
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum(labels_sub * clusters_distance)
        else:
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum((labels_sub - degree_matrix) * clusters_distance)
        self.cost = self.cost_adj - FLAGS.beta * self.cost_mod

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):

    """
    Optimizer for VGAE
    """

    def __init__(self, preds, labels, degree_matrix, model, num_nodes, pos_weight, norm, clusters_distance):

        preds_sub = preds
        labels_sub = labels

        # ELBO term (as in tkipf/gae)
        self.cost_adj = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                                                       labels = labels_sub,
                                                                                       pos_weight = pos_weight))
        self.log_lik = self.cost_adj
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost_adj -= self.kl

        # Modularity-inspired term
        if FLAGS.simple or FLAGS.fastgae: # simpler proxy of the modularity
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum(labels_sub * clusters_distance)
        else:
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum((labels_sub - degree_matrix) * clusters_distance)
        # Note: here, self.cost_adj corresponds to -ELBO. By minimizing (-ELBO-FLAGS.beta*self.cost_mod),
        # we actually maximize (ELBO+FLAGS.beta*self.cost_mod) as in the paper
        self.cost = self.cost_adj - FLAGS.beta * self.cost_mod

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
