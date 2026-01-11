import copy
import logging
import numpy as np
from overrides import overrides

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.async_aggregator import AsyncAggregator


class ODCFedBuffAggregator(AsyncAggregator):
    """
    Subclass of AsyncAggregator that implements FedBuff with 
    Orthogonal Drift Correction (ODC) to handle stragglers.
    """

    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.
        Implements Orthogonal Drift Correction.
        """
        client_id = results['client_id']
        model_version = self.client_task_model_version[client_id]
        staleness = self.round - model_version

        # 1. Retrieve the Stale Model (w_tau)
        if staleness < len(self.model_cache):
            stale_model = self.model_cache[staleness]
            logging.info(f"Found stale model in cache (Staleness = {staleness})")
        else:
            # Fallback for extreme staleness
            stale_model = self.model_cache[-1]
            logging.warning(f"Stale model not found! Using oldest. (Staleness = {staleness})")

        # 2. Get Current Global Model (w_t)
        current_weights = self.model_wrapper.get_weights()

        # 3. Process Client Weights with Orthogonal Drift Correction
        client_weights = results['update_weight']
        if isinstance(client_weights, dict):
            client_weights = [x for x in client_weights.values()]
        
        # Calculate Scaling Factor
        base_dampening = 1.0 / (1.0 + staleness) ** 0.5
        
        processed_delta = []

        # Iterate layer by layer
        for w_c, w_tau, w_t in zip(client_weights, stale_model, current_weights):
            
            # Calculate Raw Client Delta (Delta = w_c - w_tau)
            client_delta = w_c - w_tau
            
            # Calculate Global Drift (Drift = w_t - w_tau)
            drift = w_t - w_tau
            drift_norm = np.linalg.norm(drift)
            
            # Handle Zero Drift Edge Case (mostly for 0 Staleness)
            if drift_norm < 1e-7:
                # If no drift exists, use standard scaling.
                processed_delta.append(client_delta * base_dampening)
                logging.info(f"Drift Norm close to 0, skipping ODC processing!")
                continue

            # Normalized Drift Vector
            d_hat = drift / drift_norm
            
            # Project Client Delta onto Drift Direction
            parallel_mag = np.sum(client_delta * d_hat)
            
            v_parallel = parallel_mag * d_hat
            v_orthogonal = client_delta - v_parallel
            
            # Apply Selective Dampening
            beta_odc = 1.0 
            alpha_odc = base_dampening

            # Recombine Components
            corrected_layer = (alpha_odc * v_parallel) + (beta_odc * v_orthogonal)
            processed_delta.append(corrected_layer)

        # 4. Accumulate Corrected Deltas
        if self._is_first_result_in_round():
            self.model_weights = processed_delta
        else:
            self.model_weights = [acc + new for acc, new in zip(self.model_weights, processed_delta)]

        # 5. Apply Global Update
        if self._is_last_result_in_round():
            buffer_size = self.args.num_participants
            self.model_weights = [d / buffer_size for d in self.model_weights]
            
            server_lr = self.args.server_learning_rate
            server_momentum = self.args.server_momentum

            if self.server_momentum_buffer is None:
                self.server_momentum_buffer = [np.zeros_like(d) for d in self.model_weights]
            
            self.server_momentum_buffer = [
                server_momentum * m + d 
                for m, d in zip(self.server_momentum_buffer, self.model_weights)
            ]
            
            final_current_weights = self.model_wrapper.get_weights()
            self.model_weights = [w + server_lr * u for w, u in zip(final_current_weights, self.server_momentum_buffer)]
            
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
            self.aggregation_denominator = 0

if __name__ == "__main__":
    aggregator = ODCFedBuffAggregator(parser.args)
    aggregator.run()