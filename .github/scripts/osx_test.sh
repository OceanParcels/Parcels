export PATH="$HOME/miniconda/bin:$PATH"
source activate parcels

export CONDA_BUILD_SYSROOT=/
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/Applications/Xcode.app/Contents//Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/

py.test -v -s tests/
