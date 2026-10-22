Attribute VB_Name = "RunOptimizer"
' Excel button -> runs the Python optimizer (CLI). Put the .xlsm next to run_optimizer.py.
Sub RunPythonOptimizer()
    Dim cmd As String
    cmd = "cmd /c cd /d """ & ThisWorkbook.Path & """ && python run_optimizer.py > outputs_log.txt 2>&1"
    CreateObject("WScript.Shell").Run cmd, 0, True
    MsgBox "Optimizer finished - see the outputs folder and outputs_log.txt", vbInformation
End Sub
