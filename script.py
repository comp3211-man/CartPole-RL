import subprocess

while(True):
    call="python main.py --random"
    returncode = subprocess.call(call, shell=True)