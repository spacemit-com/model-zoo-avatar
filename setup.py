from pathlib import Path
import os,subprocess
from setuptools import setup,Extension,find_packages
from setuptools.command.build_ext import build_ext

ROOT=Path(__file__).resolve().parent
class CMakeBuild(build_ext):
    def build_extension(self,ext):
        import sys
        build=Path(self.build_temp).resolve()/ext.name.replace('.','_');build.mkdir(parents=True,exist_ok=True)
        dest=Path(self.build_lib).resolve()
        args=['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
              '-DBUILD_TESTING=OFF','-DBUILD_PYTHON_BINDINGS=ON','-DAVATAR_WHEEL_LAYOUT=ON',
              '-DPython3_EXECUTABLE='+sys.executable,'-DCMAKE_INSTALL_PREFIX='+str(dest)]
        for name in ['ONNXRUNTIME_INCLUDE_DIR','OPENCV_INCLUDE_DIR','pybind11_DIR']:
            if os.environ.get(name):args.append('-D'+name+'='+os.environ[name])
        if not os.environ.get('pybind11_DIR'):
            import pybind11
            args.append('-Dpybind11_DIR='+pybind11.get_cmake_dir())
        subprocess.run(args,check=True)
        subprocess.run(['cmake','--build',str(build),'--parallel',os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL','2')],check=True)
        subprocess.run(['cmake','--install',str(build)],check=True)

setup(packages=find_packages('python'),package_dir={'':'python'},
      ext_modules=[Extension('spacemit_avatar._spacemit_avatar',sources=[])],
      cmdclass={'build_ext':CMakeBuild},package_data={'spacemit_avatar':['*.so*','bin/*']})
