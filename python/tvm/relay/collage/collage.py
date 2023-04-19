# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Mostly helper methods which interface the main C++ Collage implementation with Python.
   See relay.transform.CollagePartition for the main Collage entrypoint."""

import logging
import os
import math
import tempfile
from tvm import relay
import numpy as np
from tvm.ir import IRModule
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Call, Var, Constant, TupleGetItem
import tvm
from tvm._ffi.registry import register_func, register_object
from tvm.runtime import Object
from . import _ffi_api

# Parameters to use when estimating latency (of both partitions and overall models).
MEASURE_NUMBER = 20
MEASURE_REPEAT = 5
WARMUP_MIN_REPEAT_MS = 250


@register_object("relay.collage.CostEstimator")
class CostEstimator(Object):
    """CostEstimator class"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.CostEstimator)


@register_object("relay.collage.MockCostEstimator")
class MockCostEstimator(Object):
    """MockEstimator class"""

    def __init__(self, target_costs, max_estimates=0):
        self.__init_handle_by_constructor__(_ffi_api.MockCostEstimator, target_costs, max_estimates)


@register_object("relay.collage.CustomCostEstimator")
class CustomCostEstimator(Object):
    """CustomEstimator class"""

    def __init__(self, py_fn_estimator="tvm.relay.collage.estimate_seconds_custom"):
        self.__init_handle_by_constructor__(_ffi_api.CustomCostEstimator, py_fn_estimator)


def arg_for(arg_type, device):
    """Returns a test argument of Relay arg_type on device"""
    assert isinstance(arg_type, tvm.ir.TensorType)
    return tvm.nd.array(
        np.random.uniform(-1.0, 1.0, size=arg_type.concrete_shape).astype(arg_type.dtype),
        device=device,
    )


def vm_estimate_seconds(device, the_vm, func_name, args):
    """Returns the estimated latency, in seconds, of running func_name with args on the_vm."""
    # Warmup
    the_vm.benchmark(
        device, repeat=1, number=1, min_repeat_ms=WARMUP_MIN_REPEAT_MS, func_name=func_name, **args
    )
    # One more time, with feeling
    return the_vm.benchmark(
        device,
        repeat=MEASURE_REPEAT,
        number=MEASURE_NUMBER,
        min_repeat_ms=0,
        func_name=func_name,
        **args,
    )


@register_func("tvm.relay.collage.estimate_seconds")
def estimate_seconds(mod, target):
    """Returns the mean execution time of "main" in mod on target with params. The module
    may contain "Primitive" functions, possibly with "Compiler" attributes."""
    device = tvm.device(target.get_target_device_type())

    try:
        # Build the module.
        logging.info("Compiling module to estimate")
        exe = tvm.relay.vm.compile(mod, target)
    except RuntimeError as err:
        # A build failure indicates the partition is not supported.
        # eg trying to build an nn.batch_norm on GPU, which has no schedule since we assume it
        # is only ever used with a tuple projection which is rewritten away.
        logging.info("Assigning module infinite cost since unable to build: %s", err)
        return math.inf

    # Finalize compilation
    tmp_dir = tempfile.mkdtemp()
    code, lib = exe.save()
    lib_path = os.path.join(tmp_dir, "library.so")
    # TODO(mbs): Avoid nvcc dependency?
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    lib = tvm.runtime.load_module(lib_path)
    exe = tvm.runtime.vm.Executable.load_exec(code, lib)

    # Benchmark the module.
    the_vm = tvm.runtime.vm.VirtualMachine(exe, device)
    func_name = "main"
    main_args = {v.name_hint: arg_for(v.checked_type, device) for v in mod[func_name].params}
    logging.info("Benchmarking module to estimate")
    profile = vm_estimate_seconds(device, the_vm, func_name, main_args)
    logging.info("profile: %s", profile)
    return profile.median  # seconds


def make_labelled_dfpattern_partition_rule_wrapper(compiler, pattern_tuple):
    """Returns a DFPatternPartitionRule representing one (label, pattern, predicate) entry from
    the pattern table for external codegen compiler"""
    if len(pattern_tuple) == 2:
        rule_name, dataflow_pattern = pattern_tuple
        return _ffi_api.MakeLabelledDFPatternPartitionRule(compiler, rule_name, dataflow_pattern)
    else:
        rule_name, dataflow_pattern, predicate = pattern_tuple
        return _ffi_api.MakeLabelledDFPatternPartitionRuleWithPredicate(
            compiler, rule_name, dataflow_pattern, predicate
        )


@register_func("tvm.relay.collage.make_byoc_partition_rule")
def make_byoc_partition_rule(compiler):
    """Returns the PartitionRule for external codegen compiler"""
    pattern_table = tvm.relay.op.contrib.get_pattern_table(compiler)
    assert (
        pattern_table is not None
    ), f"No pattern table entry was found for BYOC compiler {compiler}"
    logging.info(
        "Converting %s rules for %s for use in pattern style BYOC lowering/codegen",
        len(pattern_table),
        compiler,
    )
    sub_rules = [
        make_labelled_dfpattern_partition_rule_wrapper(compiler, pattern_tuple)
        for pattern_tuple in pattern_table
    ]
    return _ffi_api.MakePatternBYOCPartitionRule(compiler, sub_rules)


@register_func("relay.backend.collage.rewrite_io_layout")
def rewrite_io_layout(config, mod):
    """ Return single layout graph by removing input/output 
        layout transform if input/output differs with graph layout """
    class RewriteInLayout(ExprMutator):
        def __init__(self):
            super().__init__()
            self.input_vars = []

        def visit_call(self, call):
            new_fn = self.visit(call.op)
            if (
                isinstance(call.op, tvm.ir.expr.GlobalVar) == False
                and call.op.name == "layout_transform"
                and isinstance(call.args[0], Var)
                and call.args[0].name_hint in self.input_vars
            ):
                return relay.var(call.args[0].name_hint, call.checked_type)
            
            args = [self.visit(arg) for arg in call.args]

            return Call(new_fn, args, call.attrs)

        def __call__(self, mod):
            for p in mod["main"].params:
                self.input_vars.append(p.name_hint)
            func = self.visit(mod["main"].body)
            mod = IRModule.from_expr(func)
            return mod

    class RewriteOutLayout(ExprMutator):
        def __init__(self):
            super().__init__()

        def visit_call(self, call):
            if (isinstance(call.op, tvm.ir.expr.GlobalVar) == False
                and call.op.name == "layout_transform"):

                return call.args[0]

            return call 

        def __call__(self, mod):
            func = self.visit(mod["main"])
            mod = IRModule.from_expr(func.body)
            return mod

    new_mod = relay.transform.DefuseOps()(mod)
    print(new_mod)
    new_mod = RewriteInLayout()(new_mod)
    new_mod = RewriteOutLayout()(new_mod)
    print(new_mod)
    new_mod = relay.transform.FuseOps()(relay.transform.InferType()(new_mod))
    mod["main"] = new_mod["main"]
    mod = relay.transform.PlanDevices(config)(mod)
    return mod