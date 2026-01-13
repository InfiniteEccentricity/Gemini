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
        # We don't call get_weights() here because model_wrapper is still None
        self.hidden_weights = None 
        self.quant_bits = getattr(args, 'quant_bits', 4)
        logging.info(f"QAFeL Aggregator initialized with {self.quant_bits} bits quantization.")

    @overrides
    def update_weight_aggregation(self, results):
        """Standard FedBuff delta accumulation with QaFEL jump quantization."""
        
        # Lazy initialization of hidden state
        if self.hidden_weights is None:
            logging.info("QAFeL: Initializing global hidden state x_hat from first result.")
            # model_wrapper is guaranteed to exist now because training has started
            self.hidden_weights = copy.deepcopy(self.model_wrapper.get_weights())

        # 1. Standard FedBuff: accumulate client results into self.model_weights
        super().update_weight_aggregation(results)

        # 2. When the buffer is full (round ends)
        if self._is_last_result_in_round():
            current_weights = self.model_wrapper.get_weights()
            ordered_keys = list(current_weights.keys())
            
            # Compute jump: (Current Global Model x - Global Hidden State x_hat)
            diff = [current_weights[k] - self.hidden_weights[k] for k in ordered_keys]
            
            # 3. Server-side Quantization Q_s 
            q_s_list = qsgd_quantize(diff, bits=self.quant_bits)
            q_s_dict = {k: q_s_list[i] for i, k in enumerate(ordered_keys)}
            
            # 4. Update global hidden state: x_hat = x_hat + Q_s(x - x_hat) 
            for k in ordered_keys:
                self.hidden_weights[k] += q_s_dict[k]
            
            # 5. Broadcast: Set model_weights to the delta q_s
            # This ensures the Executor receives the quantized update
            self.model_weights = q_s_dict
            
            logging.info(f"QAFeL Round {self.round} complete. Quantized delta ready for broadcast.")

if __name__ == "__main__":
    aggregator = QAFeLAggregator(parser.args)
    aggregator.run()