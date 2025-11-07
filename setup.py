import glob
import os
import re
from setuptools import find_packages, setup, Extension
import sys, os, shutil

def _bootstrap_cuda_env():
    # If CUDA_HOME env is valid, keep it
    cuda_env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_env and os.path.exists(os.path.join(cuda_env, "bin", "nvcc")):
        return
    # Build candidate prefixes: current conda env (sys.prefix), CONDA_PREFIX, nvcc from PATH
    candidates = []
    for base in [os.environ.get("CONDA_PREFIX"), sys.prefix]:
        if base:
            candidates.extend([base, os.path.join(base, "targets", "x86_64-linux")])
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.append(os.path.dirname(os.path.dirname(nvcc_path)))
    # Choose the first containing bin/nvcc
    for base in candidates:
        if base and os.path.exists(os.path.join(base, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = base
            os.environ["CUDA_PATH"] = base
            # Ensure PATH/LD_LIBRARY_PATH include CUDA
            os.environ["PATH"] = os.path.join(base, "bin") + os.pathsep + os.environ.get("PATH", "")
            libs = [os.path.join(base, "lib64"), os.path.join(base, "lib"), os.path.join(base, "targets", "x86_64-linux", "lib")]
            ld = os.pathsep.join([p for p in libs if os.path.isdir(p)])
            if ld:
                old_ld = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ld + (os.pathsep + old_ld if old_ld else "")
            break

_bootstrap_cuda_env()

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension, CUDA_HOME)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()

def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}
    include_dirs = []
    library_dirs = []
    runtime_library_dirs = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda

        cuda_home_candidates = [
            os.environ.get('CUDA_HOME'),
            CUDA_HOME,
            os.environ.get('CONDA_PREFIX'),
        ]
        cuda_home = next((c for c in cuda_home_candidates if c and os.path.isdir(c)), None)

        if cuda_home:
            candidate_includes = [
                os.path.join(cuda_home, 'include'),
                os.path.join(cuda_home, 'targets', 'x86_64-linux', 'include'),
            ]
            candidate_libs = [
                os.path.join(cuda_home, 'lib64'),
                os.path.join(cuda_home, 'lib'),
                os.path.join(cuda_home, 'targets', 'x86_64-linux', 'lib'),
            ]

            for path in candidate_includes:
                if os.path.isdir(path) and path not in include_dirs:
                    include_dirs.append(path)

            for path in candidate_libs:
                if os.path.isdir(path) and path not in library_dirs:
                    library_dirs.append(path)
                    if path not in runtime_library_dirs:
                        runtime_library_dirs.append(path)

        for path in include_dirs:
            flag = f'-I{path}'
            if flag not in extra_compile_args['cxx']:
                extra_compile_args['cxx'].append(flag)
            if 'nvcc' in extra_compile_args:
                extra_compile_args['nvcc'].extend(['-I', path])
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=extra_compile_args)
def make_cpp_ext(name, module, sources,compiler_directives):
    return Extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        compiler_directives=compiler_directives)

def get_extensions():
    extensions = [
    make_cuda_ext(
        name='sort_vertices',
        module='src.ops.sort_vertices',
        sources=['sort_vert.cpp'],
        sources_cuda=[
            'sort_vert_kernel.cu'
        ]),
    make_cpp_ext(
        name='crdp',
        module='src.ops.crdp',
        sources=['crdp.pyx'],
        compiler_directives={
                               'language_level': 3
                           }
    )
    ]
    return extensions


setup(
    name='latentdriver',
    version="1.0",
    keywords='Automous driving planning',
    license='Apache-2.0',
    packages=find_packages(),
    include_package_data=True,
    author='Xiao, lingyu',
    author_email='lyhsiao@seu.edu.cn',
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False)