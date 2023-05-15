@echo off
REM Erstes Python-Skript ausführen
"C:\Users\widde\anaconda3\envs\tf2\python.exe" C:\Users\widde\Desktop\PFD\Object_Detection\Detect_images.py -m C:\Users\widde\Desktop\PFD\Object_Detection\inference_graph\saved_model -l C:\Users\widde\Desktop\PFD\Object_Detection\labelmap.pbtxt -i C:\Users\widde\Desktop\PFD\Object_Detection\test_images

REM Zweites Python-Skript ausführen
"C:\Users\widde\anaconda3\envs\tf2\python.exe" "C:\Users\widde\Desktop\PFD\Object_Detection\open_img.py"

REM Batch-Skript beenden
pause