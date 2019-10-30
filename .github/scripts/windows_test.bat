SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
call activate parcels
py.test -v -s tests/
parcels_get_examples examples/
py.test -v -s --nbval-lax examples/ -k "not documentation" 
