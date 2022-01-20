# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math
import os
import time
from collections import defaultdict

import torch
from packaging import version
from torch import nn
from torch.utils.data import Dataset

from .deepspeed import is_deepspeed_zero3_enabled
from .trainer import Trainer
from .trainer_utils import PredictionOutput, speed_metrics
from .utils import logging
from .debug_utils import DebugOption
from .file_utils import is_torch_tpu_available
from .decoding import decode

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(
        self, *args, untokenized_eval_dataset=None, data_args=None, output_dir: Optional[str] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._untokenized_eval_dataset = untokenized_eval_dataset
        self._max_length = data_args.val_max_target_length
        self._num_beams = data_args.num_beams
        self._output_dir = output_dir
        self._data_args = data_args
        self.mock_predictions_to_assign_zero_metric_score = self.tokenizer.encode("TOO_MANY_INPUT_TOKENS",return_tensors="np")[0]

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        untokenized_eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if max_length is not None:
            self._max_length = max_length
        if num_beams is not None:
            self._num_beams = num_beams

        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        untokenized_eval_dataset = (
            self._untokenized_eval_dataset if untokenized_eval_dataset is None else untokenized_eval_dataset
        )
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            eval_loop_output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        metrics = eval_loop_output.metrics

        if eval_loop_output.predictions is not None:
            eval_preds = self._post_process_function(untokenized_eval_dataset, eval_loop_output.predictions)

            if self._output_dir is not None and self.is_world_process_zero():
                predictions = decode(eval_preds[0], self.tokenizer, self._data_args)
                output_prediction_file = os.path.join(
                    self._output_dir, f"generated_predictions_eval_{self.state.global_step}.json"
                )
                with open(output_prediction_file, "w") as writer:
                    json.dump(predictions, writer, indent=4)

                output_labels_file = os.path.join(
                    self._output_dir, f"eval_labels.json"
                )
                if not os.path.isfile(output_labels_file) :
                    with open(output_labels_file, "w") as writer:
                        json.dump(eval_preds[1], writer, indent=4)

            if self.compute_metrics is not None:
                metrics.update(self.compute_metrics(*eval_preds))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=eval_loop_output.num_samples,
                num_steps=math.ceil(eval_loop_output.num_samples / total_batch_size),
            )
        )

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return eval_loop_output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        if max_length is not None or not hasattr(self, "_max_length"):
            self._max_length = max_length
        if num_beams is not None or not hasattr(self, "_num_beams"):
            self._num_beams = num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        if has_labels:
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        else:
            loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _post_process_function(self, untokenized_eval_dataset, predictions):
        id_to_prediction = {}
        id_to_label_ids = defaultdict(list)

        assert len(untokenized_eval_dataset) == len(self.eval_dataset)

        for i, (instance, not_valid_for_eval) in enumerate(zip(untokenized_eval_dataset, self.eval_dataset["not_valid_for_eval"])):
            if not_valid_for_eval:
                id_to_prediction[instance["id"]] = self.mock_predictions_to_assign_zero_metric_score
            else:
                id_to_prediction[instance["id"]] = predictions[i]

            if "outputs" in instance:
                id_to_label_ids[instance["id"]] = instance["outputs"]
            else:
                id_to_label_ids[instance["id"]].append(instance["output"])

        return id_to_prediction, id_to_label_ids
