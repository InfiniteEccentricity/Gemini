import copy
import logging
import numpy as np
from fedscale.cloud.execution.executor import Executor
from fedscale.utils.quantizer import qsgd_quantize
from fedscale.cloud.fllibs import *

class QAFeLExecutor(Executor):
    def __init__(self, args):
        super(QAFeLExecutor, self).__init__(args)
        
        # 1. Initialize hidden state (x_hat) with starting weights
        # We use the model_adapter to get weights as a list of tensors
        self.hidden_weights = copy.deepcopy(self.model_adapter.get_weights())
        self.quant_bits = getattr(args, 'quant_bits', 4)
        logging.info(f"QAFeL Executor initialized with {self.quant_bits} bits quantization.")

    def UpdateModel(self, model_weights):
        """
        In QaFEL, the 'model_weights' received from the server is actually q_s (quantized delta).
        We update our hidden state x_hat = x_hat + q_s.
        """
        q_s_update = model_weights
        self.round += 1
        
        # 2. Update shared hidden state x_hat using received q_s
        self.hidden_weights = [hw + qs for hw, qs in zip(self.hidden_weights, q_s_update)]
        
        # 3. Set the actual model used for training to match the updated hidden state
        self.model_adapter.set_weights(self.hidden_weights, is_aggregator=False)
        logging.info(f"QAFeL: Hidden state updated for round {self.round}")

    def training_handler(self, client_id, conf, model):
        """
        Perform local training and then quantize the delta.
        """
        # 4. Train normally starting from the hidden state (model)
        # Note: model here is the hidden_weights we synced in UpdateModel
        train_res = super(QAFeLExecutor, self).training_handler(client_id, conf, model)
        
        # 5. Extract trained weights (y_p)
        client_weights = train_res['update_weight']
        
        # Handle dictionary weights (standard in some FedScale versions)
        if isinstance(client_weights, dict):
            # Ensure order matches hidden_weights
            ordered_keys = sorted(client_weights.keys())
            client_weights_list = [client_weights[k] for k in ordered_keys]
            
            # Calculate Delta: (y_p - x_hat)
            diff = [cw - hw for cw, hw in zip(client_weights_list, self.hidden_weights)]
            
            # 6. Client-side Quantization Q_c
            q_c = qsgd_quantize(diff, bits=self.quant_bits)
            
            # Reconstruct dict for aggregator
            train_res['update_weight'] = {k: q_c[i] for i, k in enumerate(ordered_keys)}
        else:
            # Handle list weights
            diff = [cw - hw for cw, hw in zip(client_weights, self.hidden_weights)]
            train_res['update_weight'] = qsgd_quantize(diff, bits=self.quant_bits)
        
        return train_res