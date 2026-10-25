# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from transformers.testing_utils import is_torch_available, torch_device


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DynamicCache, PreTrainedModel
    from transformers.integrations.heterogeneity import SkipDescriptor, get_heterogeneous_modeling_specs
    from transformers.integrations.heterogeneity.cache_utils import _get_cache_layer_caches
    from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
        KEY_VALUE_CACHE,
        LINEAR_ATTENTION_CACHE,
    )
    from transformers.integrations.heterogeneity.skip_utils import _get_member_path


def _has_members(layer: "nn.Module", skip_descriptor: "SkipDescriptor") -> bool:
    """Return whether `layer` has a member at every path in `skip_descriptor`, ignoring the classes in its member
    keys."""
    # Classes don't count, so that a layer whose member has a class that the descriptor doesn't cover still gets a
    # subtest, whose build then fails
    for member_key in skip_descriptor:
        try:
            layer.get_submodule(_get_member_path(member_key))
        except AttributeError:
            return False
    return True


def _get_layer_indices_by_skip_type(model: "PreTrainedModel") -> dict[tuple[type, str], list[int]]:
    """For each skip type of each spec, find the layers of the spec's `layer_cls` that have all the members that the
    skip type replaces.

    Returns a dict that maps `(layer_cls, skip_type)` to the indices of those layers."""
    specs = get_heterogeneous_modeling_specs(model)
    # Models build their layers in order, so a layer's position among them is its index
    layers = [module for module in model.modules() if isinstance(module, tuple(spec.layer_cls for spec in specs))]

    layer_indices_by_skip_type = {}
    for spec in specs:
        for skip_type, skip_descriptor in spec.skip_descriptors.items():
            layer_indices_by_skip_type[spec.layer_cls, skip_type] = [
                layer_idx
                for layer_idx, layer in enumerate(layers)
                if isinstance(layer, spec.layer_cls) and _has_members(layer, skip_descriptor)
            ]
    return layer_indices_by_skip_type


def _prefill_cache(model: "PreTrainedModel", input_ids: "torch.Tensor") -> "DynamicCache":
    """Run `input_ids` through `model` with a cache, and return the cache.

    The cache is built from the model's config, so it leaves out a layer's cache when the layer's skips replace every
    member that `cache_writers` lists as writing it."""
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    return cache


def _get_unwritten_caches(cache: "DynamicCache") -> list[set[str]]:
    """Return, for each layer, the caches that `cache` holds for the layer but that are still empty."""
    unwritten_caches = []
    for layer_idx, cache_layer in enumerate(cache.layers):
        caches = _get_cache_layer_caches(type(cache_layer))
        if KEY_VALUE_CACHE in caches and cache.get_seq_length(layer_idx) > 0:
            caches.remove(KEY_VALUE_CACHE)
        if LINEAR_ATTENTION_CACHE in caches and cache.has_previous_state(layer_idx):
            caches.remove(LINEAR_ATTENTION_CACHE)
        unwritten_caches.append(caches)
    return unwritten_caches


class HeterogeneousModelingTesterMixin:
    """Tests that a model's generic heterogeneous modeling specs match the model.

    The tests check the specs, not the heterogeneity mechanism, which the other tests in this directory cover. Because
    the mixin is mixed into [`ModelTesterMixin`], the tests run for every model and skip the models without specs. They
    build the first model class, usually the base model, from the model tester's config, so they check only the kinds
    of layers that a model built from this config has.

    Expected attributes provided by [`ModelTesterMixin`]: `all_model_classes` and `model_tester`.
    """

    def _skip_if_no_heterogeneous_modeling_specs(self):
        # The specs can come from the model class or from the config's model type, so finding them takes a model
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = self.all_model_classes[0](config)
        if not get_heterogeneous_modeling_specs(model):
            self.skipTest(reason=f"{type(model).__name__} has no generic heterogeneous modeling specs")

    def _build_heterogeneous_model(self, per_layer_config: dict) -> "PreTrainedModel":
        """Build the first model class from the tester's config with `per_layer_config`, and assert that generic
        heterogeneous modeling was applied to it."""
        # Each build gets its own config, because models built with one config must agree on their skips
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.per_layer_config = per_layer_config
        model = self.all_model_classes[0](config).to(torch_device).eval()
        self.assertTrue(config.generic_heterogeneous_modeling_applied)
        return model

    def _get_input_ids(self) -> "torch.Tensor":
        _, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        return inputs_dict["input_ids"].to(torch_device)

    def test_heterogeneous_modeling_specs_build_every_layer(self):
        """Test that the model builds with its specs applied. This fails when the specs' layer classes don't cover
        every layer, or when `layer_idx_resolver` returns an invalid index or the same index for two layers."""
        self._skip_if_no_heterogeneous_modeling_specs()
        self._build_heterogeneous_model(per_layer_config={})

    def test_heterogeneous_modeling_specs_skip_descriptors(self):
        """Test that the skip descriptors match the layers of the tester's model.

        For each skip type, at least one layer must have all the members that the skip type replaces. For each such
        layer, the test builds a model that applies the skip type only in that layer, and runs a forward pass through
        it. The forward pass runs without a cache, because `test_heterogeneous_modeling_specs_cache_writers` checks
        caches. The build fails when the skip descriptor has no replacement for the class of one of the layer's
        members, and the forward pass fails when a replacement returns an output that the layer can't use, such as a
        tensor where the layer expects a tuple.

        The test doesn't check the values of those outputs."""
        self._skip_if_no_heterogeneous_modeling_specs()
        model = self._build_heterogeneous_model(per_layer_config={})
        input_ids = self._get_input_ids()

        for (layer_cls, skip_type), layer_indices in _get_layer_indices_by_skip_type(model).items():
            self.assertTrue(
                layer_indices,
                f"No {layer_cls.__name__} of the tester's model has every member of skip type '{skip_type}'. Fix the "
                "member paths, or make the tester's config build a layer that has those members.",
            )
            for layer_idx in layer_indices:
                with self.subTest(skip_type=skip_type, layer_idx=layer_idx):
                    skipped_model = self._build_heterogeneous_model({layer_idx: {"skip": [skip_type]}})
                    with torch.no_grad():
                        skipped_model(input_ids, use_cache=False)

    def test_heterogeneous_modeling_specs_cache_writers(self):
        """Test that `cache_writers` matches the layers of the tester's model.

        For each skip that the skip descriptor test applies, a forward pass with a cache must write every cache that
        the model keeps for the skipped layer, apart from caches that stay empty even without skips. A mistake in
        `cache_writers` fails this check when a skip exposes it: the layer then keeps a cache that nothing writes, or a
        remaining member writes a cache that the layer no longer has. The test also fails for some mistakes in
        `attention_mask_consumers`, because a forward pass with a cache raises when a skip removes a layer's key/value
        cache but not its attention mask."""
        self._skip_if_no_heterogeneous_modeling_specs()
        model = self._build_heterogeneous_model(per_layer_config={})
        input_ids = self._get_input_ids()
        # Some models keep caches that nothing writes even without skips, so only what a skip changes counts
        unwritten_caches_without_skips = _get_unwritten_caches(_prefill_cache(model, input_ids))

        for (_, skip_type), layer_indices in _get_layer_indices_by_skip_type(model).items():
            for layer_idx in layer_indices:
                with self.subTest(skip_type=skip_type, layer_idx=layer_idx):
                    skipped_model = self._build_heterogeneous_model({layer_idx: {"skip": [skip_type]}})
                    unwritten_caches = _get_unwritten_caches(_prefill_cache(skipped_model, input_ids))[layer_idx]
                    unwritten_caches -= unwritten_caches_without_skips[layer_idx]
                    self.assertFalse(
                        unwritten_caches,
                        f"Layer {layer_idx} keeps its {sorted(unwritten_caches)} cache, but nothing writes it, so "
                        "`cache_writers` misses a member that the skip replaced, or lists a remaining member that "
                        "doesn't write this cache.",
                    )
