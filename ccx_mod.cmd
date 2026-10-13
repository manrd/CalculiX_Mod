@echo off
setlocal
set "LIBROOT=%~dp0.mamba\e\Library"
set "PATH=%LIBROOT%\bin;%LIBROOT%\mingw-w64\bin;%PATH%"
if "%OMP_NUM_THREADS%"=="" set "OMP_NUM_THREADS=12"
if "%MKL_NUM_THREADS%"=="" set "MKL_NUM_THREADS=%OMP_NUM_THREADS%"
rem Thread affinity / MKL behavior: keep threads pinned, no dynamic shrink,
rem no migration across cores (safe HPC defaults, overridable by user env).
if "%MKL_DYNAMIC%"=="" set "MKL_DYNAMIC=FALSE"
if "%KMP_AFFINITY%"=="" set "KMP_AFFINITY=compact,granularity=fine"
if "%OMP_PROC_BIND%"=="" set "OMP_PROC_BIND=CLOSE"
"%~dp0src\ccx_mod.exe" %*
