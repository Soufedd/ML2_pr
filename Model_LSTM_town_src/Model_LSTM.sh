#!/bin/bash

python3 area_town_script.py

python3 FormatDataPrivate.py
python3 FormatDataHdb.py

python3 LSTM_hdb.py
python3 LSTM_private.py

python3 HDB.py
python3 Private.py

python3 concat_script.py