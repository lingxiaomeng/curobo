# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library

# CuRobo
from curobo_floating_base._src.util.logging import log_warn

log_warn(
    "The module 'curobo_floating_base.types.base' is deprecated and will be removed in a future version. "
    "Please use 'curobo_floating_base.types.tensor' instead."
)
