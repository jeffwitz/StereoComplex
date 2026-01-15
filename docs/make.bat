@ECHO OFF

pushd %~dp0

REM Prefer the repository venv when available (repo root: ..\.venv)
if exist "..\\.venv\\Scripts\\python.exe" (
  set SPHINXBUILD=..\\.venv\\Scripts\\python.exe -m sphinx
) else (
  set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
