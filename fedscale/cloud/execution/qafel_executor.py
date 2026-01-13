import copy
import logging
from overrides import overrides
from fedscale.cloud.execution.executor import Executor
from fedscale.utils.quantizer import qsgd_quantize
import fedscale.cloud.config_parser as parser

class QAFeLExecutor(Executor):
    def __init__(self, args):
        super().__init__(args)
        # Initialize hidden state
        weights = self.model_adapter.get_weights()
        self.hidden_weights = copy.deepcopy(list(weights.values()) if isinstance(weights, dict) else weights)
        self.quant_bits = getattr(args, 'quant_bits', 4)

    @overrides
    def UpdateModel(self, model_weights):
        """Receive quantized delta q_s and update local hidden state."""
        q_s_update = model_weights 
        self.round += 1
        
        # Ensure q_s is a list for the zip operation
        q_s_list = list(q_s_update.values()) if isinstance(q_s_update, dict) else q_s_update
        
        # Update local hidden state: x_hat = x_hat + q_s
        self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s_list)]
            
        # Sync the training model to start from the updated hidden state
        # (model_adapter.set_weights handles dicts or lists automatically)
        self.model_adapter.set_weights(self.hidden_weights, is_aggregator=False)
        logging.info(f"QAFeL Executor: Hidden state updated for round {self.round}")

    @overrides
    def training_handler(self, client_id, conf, model):
        """Train and return client quantized delta Q_c(y_p - x_hat)."""
        # 1. Run standard training
        train_res = super().training_handler(client_id, conf, model)
        
        # 2. Extract trained weights (y_p)
        y_p_raw = train_res['update_weight']
        y_p = list(y_p_raw.values()) if isinstance(y_p_raw, dict) else y_p_raw
        
        # 3. Calculate Delta: (y_p - x_hat)
        diff = [cw - hw for cw, hw in zip(y_p, self.hidden_weights)]
        
        # 4. Client-side Quantization Q_c 
        q_c = qsgd_quantize(diff, bits=self.quant_bits)
        
        # 5. Return results (match original format: dict if input was dict)
        if isinstance(y_p_raw, dict):
            keys = list(y_p_raw.keys())
            train_res['update_weight'] = {k: q_c[i] for i, k in enumerate(keys)}
        else:
            train_res['update_weight'] = q_c
        
        return train_res

if __name__ == "__main__":
    executor = QAFeLExecutor(parser.args)
    executor.run()