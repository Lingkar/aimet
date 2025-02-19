# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import torch
import json
from aimet_common.defs import MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper
from aimet_torch.examples.test_models import ModelWithTwoInputs, ModelWithTransposeConv
from aimet_torch.qc_quantize_op import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
import libpymo


class TestPerChannelQcQuantizeOpStaticGrid:
    def test_per_channel_symmetric_qdq(self):
        """ Test tensor quantizer symmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 3.81
            encodings[index].min = -3.84
            encodings[index].delta = 0.03
            encodings[index].offset = -128

        encodings[3].bw = 8
        encodings[3].max = 6.35
        encodings[3].min = -6.4
        encodings[3].delta = 0.05
        encodings[3].offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]],
                                  dtype=torch.float32)

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-6.4, -5, -3, 0, .1, 2.5]],
                                    dtype=torch.float32)
        assert torch.allclose(quant_out, expected_out, atol=1e-5)

    def test_per_channel_asymmetric_qdq(self):
        """ Test tensor quantizer asymmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 1.9999956
            encodings[index].min = -2.9999934
            encodings[index].delta = 0.0196078
            encodings[index].offset = -153

        encodings[3].bw = 8
        encodings[3].max = 2.404693
        encodings[3].min = -5.995262
        encodings[3].delta = 0.032941
        encodings[3].offset = -182

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]])

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-5.9953, -5.0070, -2.9976, 0, .09888, 2.4047]],
                                    dtype=torch.float32)
        assert torch.allclose(quant_out, expected_out, atol=0.0001)

    def test_per_channel_symmetric_compute_encodings(self):
        """ Test tensor quantizer symmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert quantizer.encoding[0].max == 7
        assert round(quantizer.encoding[0].min, 2) == -7.06

        assert quantizer.encoding[3].max == 5
        assert round(quantizer.encoding[3].min, 2) == -5.04

    def test_per_channel_asymmetric_compute_encodings(self):
        """ Test tensor quantizer asymmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert round(quantizer.encoding[0].max, 3) == 2.496
        assert round(quantizer.encoding[0].min, 3) == -7.004

        assert round(quantizer.encoding[3].max, 3) == 2.004
        assert round(quantizer.encoding[3].min, 3) == -4.996

    # -------------------------------------------
    def test_model_with_two_inputs_per_channel(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        assert isinstance(sim.model.fc2.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.fc2.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.fc2.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        assert len(sim.model.conv1_a.param_quantizers['weight'].encoding) == 10
        assert len(sim.model.fc2.param_quantizers['weight'].encoding) == 10

        model(*dummy_input)

        # Check that different encodings are computed for different channels
        assert sim.model.conv1_a.param_quantizers['weight'].encoding[0] != \
               sim.model.conv1_a.param_quantizers['weight'].encoding[1]
        assert sim.model.fc2.param_quantizers['weight'].encoding[0] != \
               sim.model.fc2.param_quantizers['weight'].encoding[1]

        sim.export('./data/', 'two_input_model_per_channel', dummy_input)

        with open("./data/two_input_model_per_channel.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
        assert len(encodings['param_encodings']) == 10
        assert len(encodings['param_encodings']['conv1_a.bias']) == 1
        assert len(encodings['param_encodings']['conv1_a.weight']) == 10
        assert encodings['param_encodings']['conv1_a.weight'][1]['bitwidth'] == 8
        assert encodings['param_encodings']['conv1_a.weight'][1]['is_symmetric'] == 'False'

    def test_set_and_freeze_param_encoding_per_channel(self):
        """ Test set and freeze parameter encoding for per-channel encodings """
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quant_module = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                              quant_scheme=QuantScheme.post_training_tf_enhanced,
                                              data_type=QuantizationDataType.int)
        quant_module.enable_per_channel_quantization()

        param_encodings = {'conv1.weight': [{'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038}
                                            ]}

        quant_module.set_and_freeze_param_encoding('conv1', param_encodings)

        assert len(quant_module.param_quantizers['weight'].encoding) == 4
        assert quant_module.param_quantizers['weight'].encoding[0].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[0].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[0].delta == 0.038
        assert quant_module.param_quantizers['weight'].encoding[3].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[3].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[3].delta == 0.038

        assert not quant_module.param_quantizers['weight'].use_symmetric_encodings
        assert quant_module.param_quantizers['weight'].bitwidth == 4

        # Reset encoding, Since encoding are frozen they should not be None after reset encoding
        quant_module.reset_encodings()

        assert len(quant_module.param_quantizers['weight'].encoding) == 4
        assert quant_module.param_quantizers['weight'].encoding[0].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[0].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[0].delta == 0.038
        assert quant_module.param_quantizers['weight'].encoding[3].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[3].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[3].delta == 0.038

    # -------------------------------------------
    def test_model_with_two_inputs_per_channel_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    # -------------------------------------------

    def test_model_with_two_inputs_per_channel_fp16_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input, default_output_bw=16, default_param_bw=16,
                                   default_data_type=QuantizationDataType.float)

        for wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    def test_model_transposed_conv_per_channel_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTransposeConv()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    def test_transposed_conv_layer_per_channel(self):
        """Model with more than 1 input"""

        num_output_channels = 4
        layer = torch.nn.ConvTranspose2d(10, num_output_channels, kernel_size=5)
        # Fill all weight values with 1
        layer.weight.data.fill_(1.0)
        encodings = [libpymo.TfEncoding() for _ in range(num_output_channels)]
        for i in range(num_output_channels):
            layer.weight.data[:, i, :, :] *= (i+1)
            encodings[i].bw = 8
            encodings[i].max = i + 0.5
            encodings[i].min = 0
            encodings[i].delta = 0.0196078
            encodings[i].offset = -153

        quantization_wrapper = StaticGridQuantWrapper(layer, weight_bw=8, activation_bw=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                      data_type=QuantizationDataType.int)
        quantization_wrapper.enable_per_channel_quantization()

        weight_quantizer = quantization_wrapper.param_quantizers['weight']
        bias_quantizer = quantization_wrapper.param_quantizers['bias']

        assert isinstance(weight_quantizer, StaticGridPerChannelQuantizer)
        assert isinstance(bias_quantizer, StaticGridPerChannelQuantizer)
        assert len(weight_quantizer._cppOp) == num_output_channels

        weight_quantizer.update_encoding_stats(quantization_wrapper._module_to_wrap.weight)
        # Assign golden vector to encodings
        weight_quantizer.encoding = encodings
        round_mode = libpymo.RoundingMode.ROUND_NEAREST
        # Quantize Dequantize
        output = weight_quantizer.quantize_dequantize(quantization_wrapper._module_to_wrap.weight, round_mode)
        expected_output = layer.weight.data
        for i in range(num_output_channels):
            expected_output[:, i, :, :] -= 0.5

        assert torch.equal(output, expected_output)
