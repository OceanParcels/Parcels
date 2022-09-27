SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
call activate parcels
parcels_get_examples examples/;
pytest -v -s --nbval-lax -k "not documentation" examples/;
