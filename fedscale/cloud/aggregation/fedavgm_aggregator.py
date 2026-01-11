import copy
import numpy as np
from overrides import overrides
import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator

class FedAvgMAggregator(Aggregator):
    """
    Implements FedAvgM (Federated Averaging with Server Momentum).
    
    Reference:
    Hsu, T. M. H., Qi, H., & Brown, M. (2019). 
    "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification".
    arXiv preprint arXiv:1909.06335.
    """

    @overrides
    def setup_env(self):
        super().setup_env()
        self.server_momentum_buffer = None
        self.total_samples_this_round = 0

    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation using Sample-Weighted Average."""
        
        # 1. Extract Client Data
        client_weights = results["update_weight"]
        if type(client_weights) is dict:
            client_weights = [x for x in client_weights.values()]
            
        sample_count = results["trained_size"]  # n_k

        # 2. Accumulate Weighted Model: sum(w_k * n_k)
        weighted_update = [w * sample_count for w in client_weights]

        if self._is_first_result_in_round():
            self.model_weights = weighted_update
            self.total_samples_this_round = sample_count
        else:
            self.model_weights = [
                acc + new 
                for acc, new in zip(self.model_weights, weighted_update)
            ]
            self.total_samples_this_round += sample_count

        # 3. Finalize Round (Apply Momentum)
        if self._is_last_result_in_round():
            # W_avg = sum(w_k * n_k) / sum(n_k)
            averaged_weights = [
                weight / self.total_samples_this_round 
                for weight in self.model_weights
            ]

            current_weights = self.model_wrapper.get_weights()

            # Calculate Pseudo-Gradient: Delta = w_t - W_avg
            pseudo_gradient = [
                curr - avg for curr, avg in zip(current_weights, averaged_weights)
            ]

            # Update Momentum Buffer
            # v_{t+1} = beta * v_t + Delta
            beta = self.args.server_momentum
            
            if self.server_momentum_buffer is None:
                self.server_momentum_buffer = [np.zeros_like(w) for w in pseudo_gradient]

            self.server_momentum_buffer = [
                beta * v + g 
                for v, g in zip(self.server_momentum_buffer, pseudo_gradient)
            ]

            # Update Global Model (w_{t+1})
            # w_{t+1} = w_t - eta * v_{t+1}
            eta = self.args.server_learning_rate
            new_global_weights = [
                curr - eta * v 
                for curr, v in zip(current_weights, self.server_momentum_buffer)
            ]

            self.model_wrapper.set_weights(
                copy.deepcopy(new_global_weights),
                client_training_results=self.client_training_results,
            )
            self.total_samples_this_round = 0

if __name__ == "__main__":
    aggregator = FedAvgMAggregator(parser.args)
    aggregator.run()