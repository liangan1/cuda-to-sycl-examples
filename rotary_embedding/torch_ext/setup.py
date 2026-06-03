"""
Step 3: Build Script for Rotary Embedding PyTorch Extension
Following SKILLS.md build pattern
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Step 3: Set CXX=icpx (CRITICAL for SYCL)
os.environ['CXX'] = 'icpx'

setup(
    name='rotary_embedding_xpu',
    ext_modules=[
        CppExtension(
            name='rotary_embedding_xpu',
            sources=['rotary_embedding_xpu.cpp'],
            extra_compile_args={
                'cxx': [
                    '-fsycl',                    # Enable SYCL
                    '-O3',                       # Optimization
                    '-fsycl-targets=spir64',    # Target Intel GPU
                    '-std=c++17',
                ]
            },
            extra_link_args=[
                '-fsycl',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
