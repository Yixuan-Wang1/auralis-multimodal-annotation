$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
python .\server.py --dataset-root 'G:\avcs' --host 127.0.0.1 --port 8765

