import copy
import logging
import numpy as np
from overrides import overrides
from fedscale.cloud.aggregation.fedbuff_aggregator import FedBuffAggregator
from fedscale.utils.quantizer import qsgd_quantize
import fedscale.cloud.config_parser as parser

class QAFeLAggregator(FedBuffAggregator):
    def __init__(self, args):
        super().__init__(args)
        # Initialize attributes as None here; do not call get_weights() yet
        self.hidden_weights = None 
        self.quant_bits = getattr(args, 'quant_bits', 4)
        self.server_momentum_buffer = None

    @overrides
    def setup_env(self):
        """
        Overrides setup_env to ensure hidden_weights are initialized 
        AFTER the model_wrapper is created in the base class.
        """
        super().setup_env() # This initializes self.model_wrapper
        
        # Now it is safe to access model_wrapper
        logging.info("QAFeL: Initializing global hidden state x_hat")
        self.hidden_weights = copy.deepcopy(self.model_wrapper.get_weights())
        logging.info(f"QAFeL Aggregator initialized with {self.quant_bits} bits quantization.")

    @overrides
    def update_weight_aggregation(self, results):
        """Standard FedBuff delta accumulation with QaFEL jump quantization."""
        super().update_weight_aggregation(results)

        if self._is_last_result_in_round():
            current_weights = self.model_wrapper.get_weights()
            ordered_keys = list(current_weights.keys())
            
            # Compute jump: (x - x_hat)
            diff = [current_weights[k] - self.hidden_weights[k] for k in ordered_keys]
            
            # Server-side Quantization Q_s 
            q_s_list = qsgd_quantize(diff, bits=self.quant_bits)
            q_s_dict = {k: q_s_list[i] for i, k in enumerate(ordered_keys)}
            
            # Update global hidden state: x_hat = x_hat + Q_s(x - x_hat) 
            for k in ordered_keys:
                self.hidden_weights[k] += q_s_dict[k]
            
            # Update model_weights for broadcast
            self.model_weights = q_s_dict
            
            logging.info(f"QAFeL Round {self.round} complete. Quantized delta ready for broadcast.")

if __name__ == "__main__":
    aggregator = QAFeLAggregator(parser.args)
    aggregator.run()