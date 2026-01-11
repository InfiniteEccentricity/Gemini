import copy
import logging
import numpy as np
from overrides import overrides

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.async_aggregator import AsyncAggregator


class FedBuffAggregator(AsyncAggregator):
    """
    Subclass of AsyncAggregator that implements the specific Delta-based 
    aggregation logic of FedBuff with Server Momentum and Server Learning Rate.
    """

    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.
        Implements the delta-based aggregation mechanism.
        """
        client_id = results['client_id']
        model_version = self.client_task_model_version[client_id]
        staleness = self.round - model_version

        # 1. Retrieve the Stale Model (w_tau)
        if staleness < len(self.model_cache):
            stale_model = self.model_cache[staleness]
            logging.info(f"Found stale model in cache (Staleness = {staleness})")
        else:
            # Fallback for extreme staleness (edge case)
            stale_model = self.model_cache[-1]
            logging.warning(f"Stale model not found in cache! Using oldest available. (Staleness = {staleness})")

        # 2. Calculate Delta (Delta = w_client - w_tau)
        client_weights = results['update_weight']
        if isinstance(client_weights, dict):
            client_weights = [x for x in client_weights.values()]
        
        # Note: Ensure weights are on CPU/Numpy for this operation
        delta = [cw - sw for cw, sw in zip(client_weights, stale_model)]

        # 3. Apply Staleness Scaling (s_tau)
        # Using the formula: 1 / sqrt(1 + staleness)
        scaling_factor = 1.0 / (1.0 + staleness) ** 0.5
        scaled_delta = [d * scaling_factor for d in delta]

        # 4. Accumulate Deltas
        if self._is_first_result_in_round():
            self.model_weights = scaled_delta
        else:
            self.model_weights = [acc + new for acc, new in zip(self.model_weights, scaled_delta)]

        # 5. Apply Global Update (Only when Buffer is Full)
        if self._is_last_result_in_round():
            # Calculate Average Delta
            buffer_size = self.args.num_participants
            self.model_weights = [d / buffer_size for d in self.model_weights]
            
            # Apply Server Learning Rate (eta_g) and Server Momentum
            server_lr = self.args.server_learning_rate
            beta = self.args.server_momentum

            if self.server_momentum_buffer is None:
                self.server_momentum_buffer = [np.zeros_like(d) for d in self.model_weights]
            
            self.server_momentum_buffer = [
                beta * m + d 
                for m, d in zip(self.server_momentum_buffer, self.model_weights)
            ]
            
            # w_{t+1} = w_t + eta_g * momentum_vector
            current_weights = self.model_wrapper.get_weights()
            self.model_weights = [w + server_lr * u for w, u in zip(current_weights, self.server_momentum_buffer)]
            
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
            self.aggregation_denominator = 0

if __name__ == "__main__":
    aggregator = FedBuffAggregator(parser.args)
    aggregator.run()