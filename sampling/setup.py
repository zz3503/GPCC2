from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sampling',
    ext_modules=[
        CUDAExtension('furthest_point_sample', [
        'src/furthest_point_sample_api.cpp',
        'src/furthest_point_sample.cpp',
        'src/furthest_point_sample_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

