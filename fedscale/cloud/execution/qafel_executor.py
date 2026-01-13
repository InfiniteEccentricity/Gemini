import copy
import logging
from overrides import overrides
from fedscale.cloud.execution.executor import Executor
from fedscale.utils.quantizer import qsgd_quantize
import fedscale.cloud.config_parser as parser

class QAFeLExecutor(Executor):
    def __init__(self, args):
        super().__init__(args)
        # Local copy of hidden state x_hat
        self.hidden_weights = copy.deepcopy(self.model_adapter.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 4)

    @overrides
    def UpdateModel(self, model_weights):
        """Receive quantized delta q_s from server and update local hidden state."""
        # In QaFEL, the received payload is the delta q_s
        q_s_update = model_weights 
        self.round += 1
        
        ordered_keys = sorted(self.hidden_weights.keys())
        
        # Update local hidden state: x_hat = x_hat + q_s
        for k in ordered_keys:
            self.hidden_weights[k] += q_s_update[k]
            
        # Sync the training model to start from the updated hidden state
        self.model_adapter.set_weights(self.hidden_weights, is_aggregator=False)
        logging.info(f"QAFeL Executor: Hidden state updated for round {self.round}")

    @overrides
    def training_handler(self, client_id, conf, model):
        """Train and return client quantized delta Q_c(y_p - x_hat)."""
        # 1. Run standard training (TorchClient.train)
        train_res = super().training_handler(client_id, conf, model)
        
        # 2. Extract trained weights (y_p)
        client_weights = train_res['update_weight']
        ordered_keys = sorted(client_weights.keys())
        
        # 3. Calculate Delta: (Trained weights - Local hidden state)
        diff = [client_weights[k] - self.hidden_weights[k] for k in ordered_keys]
        
        # 4. Client-side Quantization Q_c 
        q_c_list = qsgd_quantize(diff, bits=self.quant_bits)
        
        # 5. Pack quantized delta back into results
        train_res['update_weight'] = {k: q_c_list[i] for i, k in enumerate(ordered_keys)}
        
        return train_res

if __name__ == "__main__":
    executor = QAFeLExecutor(parser.args)
    executor.run()