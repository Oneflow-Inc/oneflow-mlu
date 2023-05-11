# Environment variables:
#
#  DEBUG
#    build with -O0 and -g

import os
from setuptools import find_packages, setup
from setuptools import Extension
from setuptools.dist import Distribution
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install

cmake_build_type = "Release"


def get_env(name, default=""):
    return os.getenv(name, default)


if get_env("DEBUG") in ["ON", "1"]:
    cmake_build_type = "Debug"
elif get_env("CMAKE_BUILD_TYPE") != "":
    cmake_build_type = get_env("CMAKE_BUILD_TYPE")

cwd = os.path.dirname(os.path.abspath(__file__))


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class build_ext(setuptools.command.build_ext.build_ext):
    def build_extension(self, ext):
        os.makedirs(self.build_temp, exist_ok=True)
        os.chdir(self.build_temp)

        cmake_args = ["-DCMAKE_BUILD_TYPE=" + cmake_build_type]
        cmake_args += ext.extra_compile_args

        self.spawn(["cmake", cwd] + cmake_args)

        build_args = ["--config", cmake_build_type, "--", "-j"]
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(cwd)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("build_ext")
        # clear build lib dir
        import glob
        import shutil

        for filename in glob.glob(f"{self.build_lib}/*"):
            try:
                os.remove(filename)
            except OSError:
                shutil.rmtree(filename, ignore_errors=True)
        super().run()


class install(setuptools.command.install.install):
    def finalize_options(self):
        super().finalize_options()
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

    def run(self):
        super().run()


setup(
    name="oneflow_mlu",
    version="0.0.1",
    description=("an OneFlow extension that targets to cambricon MLU device."),
    ext_modules=[
        Extension("oneflow_mlu", sources=[], extra_compile_args=["-DBUILD_PYTHON=ON"])
    ],
    install_requires=["oneflow"],
    cmdclass={"build_ext": build_ext, "build_py": build_py, "install": install},
    zip_safe=False,
    distclass=BinaryDistribution,
    package_dir={"oneflow_mlu": "python/oneflow_mlu"},
    packages=["oneflow_mlu"],
    package_data={"oneflow_mlu": ["*.so*", "*.dylib*", "*.dll", "*.lib",]},
)
