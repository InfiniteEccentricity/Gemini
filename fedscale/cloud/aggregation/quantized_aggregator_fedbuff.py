from overrides import overrides
from fedscale.cloud.fllibs import *
import numpy as np
import copy
import inspect

from fedscale.cloud.aggregation.fedbuff_aggregator import FedBuffAggregator


class QuantizedAggregation(FedBuffAggregator):
    '''Overrides FedBuffAggregator to add quantizer'''
    def quantizer(self, delta_list, bits=8):
        """
        Implements Qc (Algorithm 2, line 5) and Qs (Algorithm 1, line 13).
        Uses a stochastic unbiased quantizer as required by the paper.
        """
        if bits >= 32:
            return delta_list

        s = 2**bits - 1
        quantized_delta = []

        for x in delta_list:
            norm = np.linalg.norm(x)
            if norm == 0:
                quantized_delta.append(x)
                continue
            
            # Stochastic Rounding logic to maintain Unbiased Property
            v = np.abs(x) / norm
            scaled_v = v * s
            l = np.floor(scaled_v)
            probabilities = scaled_v - l
            rand = np.random.rand(*x.shape)
            xi = np.where(rand < probabilities, l + 1, l)
            
            # Reconstruction (Dequantization)
            q_x = np.sign(x) * norm * (xi / s)
            quantized_delta.append(q_x)
            
        return quantized_delta


    @overrides
    def update_weight_aggregation(self, results):
        if not hasattr(self, "_printed_model_trace"):
            model = self.model_wrapper.model

            print("\n=== MODEL TRACE (AGGREGATOR) ===")
            print("Model class     :", type(model))
            print("Model module    :", model.__class__.__module__)
            print("Model file path :", inspect.getfile(model.__class__))
            print("===============================\n")

            self._printed_model_trace = True
        client_id = results['client_id']
        model_version = self.client_task_model_version[client_id]
        staleness = self.round - model_version

        # 1. Get reference weights to ensure key alignment
        current_weights_dict = self.model_wrapper.get_weights()
        ordered_keys = list(current_weights_dict.keys())

        # 2. Initialize hidden_state as a dictionary if it doesn't exist
        if not hasattr(self, 'hidden_state') or self.hidden_state is None:
            self.hidden_state = copy.deepcopy(current_weights_dict)

        # 3. Retrieve and scale the Quantized Delta dictionary from client
        # NOTE: The client MUST return a dictionary of {layer_name: quantized_delta}
        delta_dict = results['update_weight']
        scaling_factor = 1.0 / (1.0 + staleness) ** 0.5
        
        # Process scaled deltas into a list for easier accumulation math
        scaled_delta_list = [delta_dict[k] * scaling_factor for k in ordered_keys]
        
        # 4. Accumulate Deltas in the Buffer
        if self._is_first_result_in_round():
            self.model_weights = scaled_delta_list
        else:
            self.model_weights = [acc + new for acc, new in zip(self.model_weights, scaled_delta_list)]

        # 5. Global Update (Algorithm execution)
        if self._is_last_result_in_round():
            buffer_size = self.args.num_participants
            avg_delta = [d / buffer_size for d in self.model_weights]
            
            server_lr = self.args.server_learning_rate
            beta = self.args.server_momentum
            
            # Initialize/Update Momentum
            if self.server_momentum_buffer is None:
                self.server_momentum_buffer = [np.zeros_like(d) for d in avg_delta]
            
            self.server_momentum_buffer = [
                beta * m + d for m, d in zip(self.server_momentum_buffer, avg_delta)
            ]
            
            # Calculate new high-precision global weights (x_{t+1}) as a list
            new_global_list = [current_weights_dict[k] + server_lr * u 
                            for k, u in zip(ordered_keys, self.server_momentum_buffer)]

            # 6. DOWNSTREAM: Quantize the jump qt = Qs(x_{t+1} - x_hat_t)
            q_bits = getattr(self.args, 'quantization_bits', 8)
            qt_list = [new_global_list[i] - self.hidden_state[k] for i, k in enumerate(ordered_keys)]
            qt_quantized = self.quantizer(qt_list, bits=q_bits)

            # 7. Update shared hidden state (x_hat_{t+1} = x_hat_t + qt)
            # We update it as a dictionary to prevent any future size mismatches
            for i, k in enumerate(ordered_keys):
                self.hidden_state[k] += qt_quantized[i]
            
            # 8. Broadcast: Send the FULL DICTIONARY to executors
            # This is the critical step to fix the RuntimeError
            self.model_wrapper.set_weights(copy.deepcopy(self.hidden_state))
            
            # Reset round counters
            self.aggregation_denominator = 0
            self.round += 1

if __name__ == "__main__":
    aggregator = QuantizedAggregation(parser.args)
    aggregator.run()
