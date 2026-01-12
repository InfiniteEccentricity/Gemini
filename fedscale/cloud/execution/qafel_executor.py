from fedscale.cloud.execution.executor import Executor
from fedscale.utils.quantizer import qsgd_quantize
import copy
import fedscale.cloud.logger.executor_logging as logger

class QAFeLExecutor(Executor):
    def __init__(self, args):
        super().__init__(args)
        logger.initiate_client_setting()
        # Local copy of hidden state 
        self.hidden_weights = copy.deepcopy(self.model_adapter.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 4)

    def UpdateModel(self, q_s_update):
        """Receive quantized update from server and update hidden state."""
        self.round += 1
        # Update shared hidden state using the received q_s 
        self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s_update)]
        # Set training to start from the updated hidden state 
        self.model_adapter.set_weights(self.hidden_weights)

    def training_handler(self, client_id, conf, model):
        """Perform training and quantize the resulting delta."""
        # 1. Train normally starting from hidden state 
        train_res = super().training_handler(client_id, conf, model)
        
        # 2. Calculate Client Delta: y_p - y_0 (where y_0 is hidden state) 
        client_weights = train_res['update_weight']
        if isinstance(client_weights, dict):
            client_weights = [x for x in client_weights.values()]
            
        diff = [cw - hw for cw, hw in zip(client_weights, self.hidden_weights)]
        
        # 3. Client-side Quantization Q_c 
        q_c = qsgd_quantize(diff, bits=self.quant_bits)
        
        # 4. Return quantized delta to server 
        train_res['update_weight'] = q_c
        return train_res