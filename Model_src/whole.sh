#!/bin/bash

python3 area_town_script.py

python3 FormatDataPrivate.py
python3 FormatDataHdb.py

python3 HDB_whole.py
python3 Private_whole.py

python3 concat_script_whole.py