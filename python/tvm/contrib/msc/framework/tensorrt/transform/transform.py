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
# pylint: disable=invalid-name
"""tvm.contrib.msc.framework.tensorrt.transform.transform"""

from typing import List

import tvm
from tvm.relax.transform import _ffi_api as relax_api
from tvm.contrib.msc.core.utils import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def TransformTensorRT(
    version: List[int] = None, linear_to_conv: bool = False
) -> tvm.ir.transform.Pass:
    """Transform the Function to fit TensorRT.

    Parameters
    ----------
    version: list<int>
        The tensorrt version.
    linear_to_conv: bool
        Whether to cast linear to conv2d

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """

    config = {
        "version": version or msc_utils.get_version(MSCFramework.TENSORRT),
        "linear_to_conv": linear_to_conv,
    }
    return relax_api.TransformTensorRT(msc_utils.dump_dict(config))  # type: ignore
