$Logfile = "C:\Internal\deep_events\deep-events\scheduled_train\scheduled_train.log"
function WriteLog
{
Param ([string]$LogString)
$Stamp = (Get-Date).toString("yyyy/MM/dd HH:mm:ss")
$LogMessage = "$Stamp $LogString"
Add-content $LogFile -value $LogMessage
}

WriteLog "Training Start"

C:\Internal\deep_events\.env\scripts\activate.ps1
python C:\Internal\deep_events\deep-events\scheduled_train\scheduled_train.py

WriteLog "Training End"
Add-content $LogFile -value "-----------------------------------------"