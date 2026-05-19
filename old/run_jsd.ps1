# Using the 'py' launcher is recommended on Windows
# as it will find the installed Python version automatically.
.venv\Scripts\activate  
Start-Process powershell -ArgumentList "-NoExit -Command & {
       # Your script here
       python jsd.py $alg mnist
   }" -WindowStyle Hidden
