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
        self.hidden_weights = None 
        self.quant_bits = getattr(args, 'quant_bits', 4)
        logging.info(f"QAFeL Aggregator initialized with {self.quant_bits} bits quantization.")

    @overrides
    def update_weight_aggregation(self, results):
        # Lazy initialization of hidden state
        if self.hidden_weights is None:
            weights = self.model_wrapper.get_weights()
            # Convert dict to list if necessary for consistent internal math
            self.hidden_weights = copy.deepcopy(list(weights.values()) if isinstance(weights, dict) else weights)

        # 1. Standard FedBuff aggregation
        super().update_weight_aggregation(results)

        # 2. Global Update Logic
        if self._is_last_result_in_round():
            # Get current high-precision global model (x)
            current_weights = self.model_wrapper.get_weights()
            x = list(current_weights.values()) if isinstance(current_weights, dict) else current_weights
            
            # Compute jump: (x - x_hat)
            diff = [cw - hw for cw, hw in zip(x, self.hidden_weights)]
            
            # 3. Server-side Quantization Q_s 
            q_s = qsgd_quantize(diff, bits=self.quant_bits)
            
            # 4. Update global hidden state: x_hat = x_hat + Q_s(x - x_hat) 
            self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s)]
            
            # 5. Broadcast: Set model_weights to the delta q_s
            # If the base class expects a dict, convert it back
            if isinstance(current_weights, dict):
                keys = list(current_weights.keys())
                self.model_weights = {k: q_s[i] for i, k in enumerate(keys)}
            else:
                self.model_weights = q_s
            
            logging.info(f"QAFeL Round {self.round} complete. Broadcast delta prepared.")

if __name__ == "__main__":
    aggregator = QAFeLAggregator(parser.args)
    aggregator.run()