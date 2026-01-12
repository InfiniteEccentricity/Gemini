import copy
from fedbuff_aggregator import FedBuffAggregator
from fedscale.utils.quantizer import qsgd_quantize

class QAFeLAggregator(FedBuffAggregator):
    def __init__(self, args):
        super().__init__(args)
        # Initialize hidden state with starting weights 
        self.hidden_weights = copy.deepcopy(self.model_wrapper.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 4)

    def update_weight_aggregation(self, results):
        """Standard FedBuff delta accumulation."""
        super().update_weight_aggregation(results)

        # When the buffer is full and global model is updated:
        if self._is_last_result_in_round():
            current_weights = self.model_wrapper.get_weights()
            
            # 1. Compute difference between global model and hidden state 
            diff = [cw - hw for cw, hw in zip(current_weights, self.hidden_weights)]
            
            # 2. Server-side Quantization Q_s 
            q_s = qsgd_quantize(diff, bits=self.quant_bits)
            
            # 3. Update hidden state: x_hat = x_hat + Q_s(x - x_hat) 
            self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s)]
            
            # NOTE: In broadcast, the server sends q_s instead of full weights 
            # This requires overriding the broadcast mechanism or payload.