#!/usr/bin/env python3
import os
import fileinput

html_files = [f for f in os.listdir() if f.split('.')[-1] == 'html']
for f in html_files:
    with fileinput.FileInput(f, inplace=True) as file:
        for line in file:
            print(line.replace('_static', 'static'), end='')
try:
    os.rename('_static', 'static')
except FileNotFoundError:
    pass
