import copy
import logging
from fedscale.cloud.aggregation.fedbuff_aggregator import FedBuffAggregator
from fedscale.utils.quantizer import qsgd_quantize

class QAFeLAggregator(FedBuffAggregator):
    def __init__(self, args):
        super().__init__(args)
        # 1. Initialize hidden state with starting weights
        # Fetching initial weights through the model_wrapper
        self.hidden_weights = copy.deepcopy(self.model_wrapper.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 4)
        logging.info(f"QAFeL Aggregator initialized with {self.quant_bits} bits")

    def update_weight_aggregation(self, results):
        """Standard FedBuff delta accumulation."""
        # This aggregates client quantized deltas into self.model_weights
        super().update_weight_aggregation(results)

        # When the buffer is full (round finishes):
        if self._is_last_result_in_round():
            current_weights = self.model_wrapper.get_weights()
            
            # 2. Compute difference (x - x_hat)
            # Ensure we are doing tensor-wise subtraction
            diff = [cw - hw for cw, hw in zip(current_weights, self.hidden_weights)]
            
            # 3. Server-side Quantization Q_s 
            q_s = qsgd_quantize(diff, bits=self.quant_bits)
            
            # 4. Update global hidden state: x_hat = x_hat + Q_s(x - x_hat) 
            self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s)]
            
            # 5. CRITICAL: Update model_weights so that 'broadcast_config' 
            # sends the updated hidden state to clients for the next round.
            self.model_weights = self.hidden_weights
            
            logging.info(f"QAFeL Round {self.round} complete. Hidden state updated and ready for broadcast.")

    # Optional: If you want to send ONLY the delta q_s instead of the full hidden state:
    def broadcast_config(self):
        config = super().broadcast_config()
        config['model'] = self.q_s  # You'd need to store q_s in self
        return config