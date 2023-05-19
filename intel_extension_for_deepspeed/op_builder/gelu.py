"""
Copyright 2021 Intel Machine Learning Performance Yuankun Shi
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class GeluBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_GELU"
    NAME = "gelu"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        return super().extra_ldflags()

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/ds_gelu_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/gelu_kernels.cpp')
        ]

    def include_paths(self):
        includes = [sycl_kernel_include('csrc/includes'), 'csrc/includes']
        return includes