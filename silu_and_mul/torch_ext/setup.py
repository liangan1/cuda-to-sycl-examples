from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# oneAPI icpx compiler with SYCL support
os.environ['CXX'] = 'icpx'

setup(
    name='silu_and_mul_xpu',
    ext_modules=[
        CppExtension(
            name='silu_and_mul_xpu',
            sources=['silu_and_mul_xpu.cpp'],
            extra_compile_args={
                'cxx': [
                    '-fsycl',
                    '-O3',
                    '-fsycl-targets=spir64',
                    '-std=c++17',
                ]
            },
            extra_link_args=['-fsycl', '-fsycl-targets=spir64'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
