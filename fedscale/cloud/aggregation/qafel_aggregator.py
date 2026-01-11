import copy
from fedscale.cloud.execution.executor import Executor
from fedscale.utils.quantizer import qsgd_quantize  # Assuming you created this utility

class QAFeLExecutor(Executor):
    def __init__(self, args):
        super().__init__(args)
        # 1. Properly initialize the hidden state from the starting model weights
        self.hidden_weights = copy.deepcopy(self.model_adapter.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 32)

    def UpdateModel(self, q_s_update):
        """
        QAFeL specific: Receive quantized DELTA q_s from the server 
        and update the shared local hidden state.
        """
        self.round += 1
        # 2. Synchronize hidden state: x_hat = x_hat + q_s
        self.hidden_weights = [
            hw + qs for hw, qs in zip(self.hidden_weights, q_s_update)
        ]
        # 3. Ensure the next local training starts from this synchronized state
        self.model_adapter.set_weights(self.hidden_weights, is_aggregator=False)

    def training_handler(self, client_id, conf, model):
        """
        QAFeL specific: Quantize the local update before sending to server.
        """
        # A. Start training from the current hidden state
        self.model_adapter.set_weights(self.hidden_weights, is_aggregator=False)
        
        # B. Call the base training logic (performs local SGD)
        train_res = super().training_handler(client_id, conf, model)
        
        # C. Calculate Client Delta: diff = y_final - x_hat
        client_weights = train_res['update_weight']
        if isinstance(client_weights, dict):
            client_weights = [x for x in client_weights.values()]
            
        diff = [cw - hw for cw, hw in zip(client_weights, self.hidden_weights)]
        
        # D. Client-side Quantization: q_c = Q_c(diff)
        q_c = qsgd_quantize(diff, bits=self.quant_bits)
        
        # E. Send the quantized DELTA back to the server
        train_res['update_weight'] = q_c
        return train_res