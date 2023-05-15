@echo off
REM Erstes Python-Skript ausführen
"C:\Users\33782\AppData\Local\Programs\Python\Python311\python.exe" D:\tello\PFD\Object_Detection\Detect_images.py -m D:\tello\PFD\Object_Detection\inference_graph\saved_model -l D:\tello\PFD\Object_Detection\labelmap.pbtxt -i D:\tello\Images

REM Zweites Python-Skript ausführen
"C:\Users\33782\AppData\Local\Programs\Python\Python311\python.exe" "D:\tello\PFD\Object_Detection\open_img.py"

REM Batch-Skript beenden
pause