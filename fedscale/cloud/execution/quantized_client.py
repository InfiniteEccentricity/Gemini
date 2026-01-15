from fedscale.cloud.execution.torch_client import TorchClient
from overrides import overrides
import torch
import numpy as np
import logging
import copy
import math
import time


class QuantizedClient(TorchClient):
    def quantizer(self, delta_list, bits=8):
        """Stochastic unbiased quantizer for Upstream communication."""
        if bits >= 32:
            return delta_list

        s = 2**bits - 1
        quantized_delta = []

        for x in delta_list:
            norm = np.linalg.norm(x)
            if norm == 0:
                quantized_delta.append(x)
                continue
            
            v = np.abs(x) / norm
            scaled_v = v * s
            l = np.floor(scaled_v)
            probabilities = scaled_v - l
            rand = np.random.rand(*x.shape)
            xi = np.where(rand < probabilities, l + 1, l)
            
            # Reconstruction (Dequantization) to float for the server to aggregate
            q_x = np.sign(x) * norm * (xi / s)
            quantized_delta.append(q_x)
            
        return quantized_delta
    
    @overrides
    def train(self, client_data, model, conf):
        client_id = conf.client_id
        logging.info(f"Start to train (CLIENT: {client_id}) ...")
        
        # --- QAFeL Algorithm 2, Line 1: y0 = starting weights ---
        # Capture the initial model weights (the hidden state received from server)
        y0 = copy.deepcopy(model.state_dict())

        model = model.to(device=self.device)
        model.train()

        original_local_steps = conf.local_steps
        if getattr(conf, 'local_epochs', None):
            steps_per_epoch = len(client_data)
            target_steps = max(1, int(conf.local_epochs * steps_per_epoch))
            conf.local_steps = target_steps
        
        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)
        
        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None
        self.total_samples_processed = 0
        
        # --- QAFeL Algorithm 2, Lines 2-4: Local SGD steps ---
        while self.completed_steps < conf.local_steps:
            try:
                self.train_step(client_data, conf, model, optimizer, criterion)
                if self.completed_steps % 5 == 0:
                    logging.info(f"--- EXECUTOR DEBUG: Client {client_id} at step {self.completed_steps} ---")
            except Exception as ex:
                error_type = ex
                break

        # --- QAFeL Algorithm 2, Line 5: Δ = Qc(y0 - yp) ---
        # yp = current state of model after training
        yp = model.state_dict()
        raw_delta = [y0[k].cpu().numpy() - yp[k].cpu().numpy() for k in y0.keys()]        
        # Calculate the raw delta (the jump)
        
        # Apply Upstream Quantization
        q_bits = getattr(self.args, 'quantization_bits', 8)
        quantized_delta = self.quantizer(raw_delta, bits=q_bits)
        quantized_delta_dict = {k: v for k, v in zip(y0.keys(), quantized_delta)}

        results = {
            'client_id': client_id, 
            'moving_loss': self.epoch_train_loss,
            'trained_size': self.total_samples_processed, 
            'success': self.completed_steps == conf.local_steps
        }

        if error_type is None:
            logging.info(f"Training of (CLIENT: {client_id}) completes")
        else:
            logging.info(f"Training of (CLIENT: {client_id}) failed as {error_type}")

        results['utility'] = math.sqrt(self.loss_squared) * float(trained_unique_samples)
        
        # We return the quantized delta instead of the full model params
        results['update_weight'] = quantized_delta_dict
        results['wall_duration'] = 0

        conf.local_steps = original_local_steps
        return results

