SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
call activate parcels
pytest -v -s tests/
