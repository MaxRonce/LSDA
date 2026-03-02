# lsda.ps1 -- project launcher that sets up the Hadoop/Java environment
# required for PySpark to work correctly on Windows.
#
# Requirements:
#   - Java 21 installed at C:\Program Files\Java\jdk-21
#   - winutils.exe at C:\hadoop\bin\winutils.exe  (auto-downloaded by setup)
#
# Usage:  .\lsda.ps1 train --framework spark --optuna --n-trials 10
#         .\lsda.ps1 eda
#         .\lsda.ps1 run-all

# Java 21 is required -- Java 23 removed Subject.getSubject() used by Hadoop
$env:JAVA_HOME   = "C:\Program Files\Java\jdk-21"
$env:PATH        = "$env:JAVA_HOME\bin;$env:PATH"

# winutils.exe is required on Windows for Hadoop filesystem operations
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH        = "$env:HADOOP_HOME\bin;$env:PATH"

& "$PSScriptRoot\.venv\Scripts\python.exe" -m lsda @args
