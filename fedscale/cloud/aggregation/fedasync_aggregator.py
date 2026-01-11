import copy
import logging
import numpy as np
from overrides import overrides

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.async_aggregator import AsyncAggregator

class FedAsyncAggregator(AsyncAggregator):
    """
    Implements FedAsync: Asynchronous Federated Optimization.
    Ref: https://arxiv.org/abs/1903.03934
    """

    def __init__(self, args):
        super().__init__(args)
        
        # Force buffer size to 1.
        if self.args.num_participants != 1:
            logging.warning(f"[FedAsync] Overriding 'num_participants' from {self.args.num_participants} to 1.")
            self.args.num_participants = 1
    

    @overrides
    def update_weight_aggregation(self, results):
        """
        Updates the global model immediately using the FedAsync rule:
        w_t = (1 - alpha_t) * w_{t-1} + alpha_t * w_client
        
        where alpha_t = alpha * s(staleness)
        """
        client_id = results['client_id']
        model_version = self.client_task_model_version[client_id]
        staleness = self.round - model_version
        
        # 1. Mixing Hyperparameter alpha_t
        base_alpha = self.args.server_learning_rate  
        
        # Staleness function s(t - tau). 
        staleness_factor = (staleness + 1) ** (-0.5)
        
        alpha_t = base_alpha * staleness_factor

        # 2. Get Client Weights (w_new)
        client_weights = results['update_weight']
        if isinstance(client_weights, dict):
            client_weights = [x for x in client_weights.values()]

        # 3. Get Current Global Weights (w_{t-1})
        current_weights = self.model_wrapper.get_weights()

        # 4. Perform Weighted Average Update
        # w_t = (1 - alpha) * w_{t-1} + alpha * w_client
        # Rearranged: w_t = w_{t-1} - alpha * (w_{t-1} - w_client)
        
        # Note: We compute (w_client - w_{t-1}) which is essentially the pseudo-gradient
        diff = [cw - gw for cw, gw in zip(client_weights, current_weights)]
        self.model_weights = [gw + alpha_t * d for gw, d in zip(current_weights, diff)]
        self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
        self.aggregation_denominator = 0

if __name__ == "__main__":
    aggregator = FedAsyncAggregator(parser.args)
    aggregator.run()