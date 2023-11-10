ECHO %DATE% %TIME% >> C:\Internal\deep_events\deep-events\local\scheduled_train_log.txt
ECHO Starting scheduled train >> C:\Internal\deep_events\deep-events\local\scheduled_train_log.txt

@REM call C:\Internal\deep_events\.env\Scripts\activate.bat
python test_scheduler.py

ECHO %DATE% %TIME% >> C:\Internal\deep_events\deep-events\local\scheduled_train_log.txt
ECHO Finished scheduled train >> C:\Internal\deep_events\deep-events\local\scheduled_train_log.txt
ECHO ------------------------  >> C:\Internal\deep_events\deep-events\local\scheduled_train_log.txt

timeout /t 10