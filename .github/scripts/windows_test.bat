SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
call activate parcels
parcels_get_examples examples/
py.test -v -s --nbval-lax -k "not documentation" tests/ examples/;
