# Parkplatzfinder-Drohne

## Vision (Inhalt und Ziele)
### Grundidee 
Bei der immer weiter steigenden Zahl an Autos wird die Parkplatzsuche immer schwierger und zeitaufwändiger. Vor allem bei grossen PArkfelder kann es vorkommen, dass man mit dem Auto minutenlang hin und herfährt, ohne eine freie Lücke zu finden. Die Folgen davon können sowohl wiederholtes Zu-Spät-Kommen zum Unterricht oder zu Arbeit wie auch Wut und Verzeiflung erfolglosen Autofahrer sein. 

Aus diesem Grund ist die Idee des Projekts, ein Drohne zu konzipieren und zu entwickeln, die den Autofahrer in der Parkplatzsuche unterstützen soll. 

### Funktionsweise

Eine Drohne soll hochfliegen und ein Bild der Umgebung machen. Weiter muss dieses analysiert werden, um frei Parkplätze zu erkennen. Diese Informationen sollen dem Fahrer vermittelt werden und ihn zum Parkplatz führen. Das System sollte kompakt gehalten werden, so dass der Fahrer dieses, ohne auszusteigen, bedienen kann 

## GitHub Inhalte 

### Flight control

#### Regelung:
Um einen Drohnenflug erfolgreich zu beenden, muss die Drohne an einem bestimmten Punkt landen. Um dies zu erreichen, verwenden wir einen PID-Regler. Der PID-Regler ist ein Feedback-Regler, der verwendet wird, um eine Zielposition zu erreichen und beizubehalten. Er nutzt das Feedback des Systems, um den Fehler zwischen der aktu-ellen Position und der Zielposition zu berechnen und dann die Steuerung entsprechend anzupassen, um den Fehler zu minimieren. Der PID-Regler besteht aus drei Komponenten: P (Proportional), I (Integral) und D (Derivative).
In diesem Projekt wird ein PID-Regler mit einem Machine-Learning-Modell kombinieren, das die Landegenauigkeit der Drohne verbessern soll. Das Ziel ist es, das Zentrum der Landeplattform zu erkennen und als Referenzpunkt für den PID-Regler zu verwenden. Der PID-Regler wird dann die erforderlichen Anpassungen vornehmen, indem er Befehle an die Motoren sendet, um sicherzustellen, dass das Zentrum (Referenzpunkt des PID-Reglers) in der Mitte des Kamerabildes bleibt. Dadurch sollte es der Drohne möglich sein, präzise auf der Landeplattform zu landen.

#### Image Erkennung und Pre-processing
Die Funktion "findLanding" ist eine Python-Funktion, die verwendet wird, um die Landeplattform der Drohne zu finden. Dabei wird ein zuvor trainiertes Machine-Learning-Modell und OpenCV verwendet, um die Plattform zu erkennen. Anschliessend gibt die Funktion die Koordinaten der erkannten Plattform zurück.
Der Code beginnt mit dem Laden des trainierten Modells, das in der Datei "my_model.h5" gespeichert ist. Das Modell wird mit der Methode "load_model" aus einem Framework namens "models" geladen.
Der nächste Schritt besteht darin, das Eingabevideo in ein Graustufenbild umzuwandeln. Dies wird mithilfe der OpenCV-Bibliothek und der Methode "fromarray" aus der PIL-Bibliothek erreicht. Das Video wird auf eine Grösse von 224x224 Pixeln skaliert, um es für das Modell geeignet zu machen. Das skalierte Bild wird dann in ein NumPy-Array konvertiert und um eine zusätzliche Dimension erweitert.
Anschliessend beginnt die Bilderkennung. Das Modell wird verwendet, um Vorhersagen für das Eingabebild zu machen. Die Ausgabe des Modells ist eine Liste von Plattformen, die im Bild erkannt wurden. Jede Plattform wird durch ihre Koordinaten (x, y) und ihre Breite und Höhe (pw, ph) repräsentiert.
Es werden auch drei leere Listen definiert: "center_platform", "area_platform" und "probabilities". Diese Listen werden verwendet, um die Koordinaten, Flächen und Wahrscheinlichkeiten der erkannten Plattformen zu spei-chern. Es folgt eine Schleife, die über alle erkannten Plattformen iteriert. Für jede Plattform wird die Wahrschein-lichkeit berechnet, dass es sich um die Landeplattform handelt. Dazu wird ein Ausschnitt des Bildes genommen, der der Grösse der Plattform entspricht, und dieser Ausschnitt wird an das Modell weitergegeben. Die zurückgeg-ebene Wahrscheinlichkeit wird der Liste "probabilities" hinzugefügt.
Zusätzlich wird ein Rechteck um die erkannte Plattform gezeichnet, um sie im Video visuell darzustellen. Die Koor-dinaten des Rechtecks werden verwendet, um den Mittelpunkt der Plattform zu berechnen, sowie die Fläche des Rechtecks. Diese Informationen werden den Listen "center_platform" bzw. "area_platform" hinzugefügt. Um den Mittelpunkt visuell darzustellen, wird ein grüner Kreis um den berechneten Mittelpunkt gezeichnet.
Nachdem alle Plattformen verarbeitet wurden, werden sie basierend auf ihren Wahrscheinlichkeiten sortiert. Die Liste "sorted_platform" enthält nun die erkannten Plattformen in absteigender Reihenfolge ihrer Wahrscheinlich-keiten.
Schliesslich wird das Video mit der markierten Plattform zurückgegeben. Wenn mindestens eine Plattform erkannt wurde, wird die Plattform mit der höchsten Wahrscheinlichkeit ausgewählt und ihre Koordinaten, Grösse und Flä-che zurückgegeben. Falls keine Plattform erkannt wurde, wird der Wert 0 zurückgegeben.


##### PID-Steurung
Der Code beginnt damit, die vorherigen Fehler in eine Liste zu konvertieren und die Informationen über die er-kannte Plattform abzurufen, einschliesslich ihrer Fläche und des Zentrums. Die X- und Y-Koordinaten sowohl des Zentrums des erkannten Objekts wie auch des Mittelpunktes des Videos werden berechnet. 
Anschliessend wird der Fehler für die X-Achse berechnet, indem der Abstand des Mittelpunkts des Videos vom Zentrum der erkannten Plattform bestimmt wird. Wenn sich das Objekt auf der linken Seite des Videos befindet, wird ein positiver Geschwindigkeitswert berechnet, der mit dem PID-Regler und dem vorherigen Fehler aktualisiert wird. Wenn sich das Objekt auf der rechten Seite des Videos befindet, wird ein negativer Geschwindigkeitswert berechnet, der mit dem PID-Regler und dem vorherigen Fehler aktualisiert wird. Wenn sich das Objekt genau in der Mitte des Videos befindet, wird keine Bewegung durchgeführt. Der gleiche Prozess wird für die Y-Achse wiederholt.
Dann wird der Fehler für die Z-Achse berechnet, indem die Differenz zwischen der Fläche des Objekts und einem Referenzwert bestimmt wird. Wenn die Fläche des erkannten Objekts kleiner ist als der Referenzwert, wird eine Abwärtsbewegung der Drohne berechnet und mit dem PID-Regler und dem vorherigen Fehler aktualisiert. Wenn die Fläche des erkannten Objekts gleich oder grösser als der Referenzwert ist, wird keine Bewegung durchgeführt und die Drohne wird zur Landung initiiert.
Schliesslich wird die Korrektur an die Drohne gesendet, indem Geschwindigkeiten für die X-, Y- und Z-Achse be-rechnet werden. Diese Werte werden dann als Parameter an die Funktion "send_rc_control" übergeben, um die Drohne entsprechend zu steuern.


### Benutzerinterface
Mit Hilfe des kivy packages wird ein Panel geöffnet und der Benutzer erhält Anweisungen. Nach dem Drücken des Startknopfs wird das Panel upgedatet und weitere Anweisungen werden angezeigt.

### USB-Kamera
Die USB-Kamera wird mit eingelesen und das Bild wird aktualisiert und so ausgegeben. Die Kamera läuft selbstständig neben dem Hauptprogramm.

#### Flight 

#### Landing 
Das Modell für die Objekterkennung basiert auf TensorFlow und Keras. Es wird mit Bildern von Drohnen trainiert, um zwischen Drohnen und Nicht-Drohnen in Echtzeit zu unterscheiden.
##### Datasets:
Das Dataset ist wie folgt aufgebaut.  Eine CSV-Datei wird geladen, die Metadaten von den Bildern enthält. An-schliessend werden die Klassenbezeichnungen in binäre Variablen codiert, um sie für das maschinelle Lernen ver-arbeiten zu können.
Eine Liste wird erstellt, um Bild- und Bounding-Box-Daten zu speichern. Die Bilddaten werden durch Mithilfe einer Iteration aus der CSV-Datei herausgelesen. Dabei wird das Bild aus dem entsprechenden Ordner geladen, auf eine Grösse von 224 x 224 Pixeln skaliert, die Pixelwerte normiert, die Mittelwertzentrierung durchgeführt und die Standardisierung der Pixelwerte vorgenommen.
Die Bounding-Box-Koordinaten werden normalisiert, um sie auf einen gemeinsamen Massstab zu bringen. Die normalisierten Bounding-Box-Koordinaten und die zugehörigen Klassenbezeichnungen werden als Tupel in der Liste gespeichert. Die Liste wird anschliessend in ein NumPy-Array konvertiert.
Das NumPy-Array wird dann in Trainings- und Testsets mit einer Ratio von 80%-20% aufgeteilt. Die Klassenbe-zeichnungen werden ebenfalls in NumPy-Arrays konvertiert. Schliesslich werden die Trainings- und Testdaten in Tensor-Objekte konvertiert, um sie in einem Tensorflow-basierten Modell zu verwenden.

##### Architektur : 
Das Modell verwendet eine vortrainierte MobileNetV2-Architektur als Basis [7], die auf dem ImageNet-Datensatz vortrainiert wurde [8]. Das Modell wird für die Fine-Tuning-Aufgabe angepasst.
Danach werden alle Schichten des vortrainierten Modells eingefroren, um das bereits erlernte Wissen zu bewah-ren und Überanpassung zu vermeiden. Durch Einfrieren der Schichten werden die Gewichtungen nicht aktualisiert, wenn das Modell auf die neuen Daten trainiert wird. Dies bedeutet, dass das vortrainierte Modell nicht neu ange-lernt wird, sondern dass es für die neue Aufgabe angepasst wird. Dies ermöglicht eine schnellere Anpassung an neue Datensätze und verbessert die Genauigkeit des Modells.
 
Ein wichtiger Aspekt dieses Codes ist die Verwendung von Aktivierungsfunktionen wie Sigmoid und ReLU. Die Re-LU-Funktion (Rectified Linear Unit) wird häufig in Deep-Learning-Modellen verwendet, da sie eine einfache und effektive Möglichkeit bietet, nicht-lineare Merkmale hinzuzufügen. Die ReLU-Funktion gibt eine Null zurück, wenn der Eingabewert negativ ist, andernfalls gibt sie den Eingabewert zurück. Dies führt zu einer effektiven Unterdrü-ckung von Rauschen und zu einer Verbesserung der Modellgenauigkeit.
Die Sigmoid-Funktion wird häufig verwendet, um die Wahrscheinlichkeiten der Vorhersagen zu berechnen. Die Sigmoid-Funktion hat einen Sättigungsbereich, der dazu beiträgt, eine stabile numerische Ausgabe zu erzeugen, und sie liefert Werte zwischen 0 und 1, was ideal für die Wahrscheinlichkeitsberechnung ist.
Der Dense-Layer ist eine Art von Schicht, die häufig in neuronalen Netzen verwendet wird. In diesem Modell wer-den zwei Dense-Layer verwendet, um die Merkmale zu reduzieren und die Vorhersagen zu generieren. Der Dense-Layer wird auch als vollständig verbundener Layer bezeichnet, da jeder Knoten im aktuellen Layer mit jedem Kno-ten im vorherigen Layer verbunden ist. Der Dense-Layer kann verwendet werden, um lineare oder nicht-lineare Operationen auf den Merkmalen durchzuführen.
Der Global Average Pooling Layer (GAP) wird verwendet, um eine globale Zusammenfassung der Feature-Maps zu erhalten. Im Wesentlichen wird der GAP-Layer verwendet, um die Feature-Maps zu einer einzigen Zahl pro Fea-ture-Map zu reduzieren. Dies hilft, die Merkmalsdimensionen zu reduzieren und die Rechenleistung zu verbessern, da weniger Parameter für die folgenden Schichten erforderlich sind.
Die Verwendung von gewichteten Schichten aus der ImageNet-Datenbank ist eine gängige Praxis in der Bildklassi-fizierung. ImageNet ist eine riesige Datenbank mit Millionen von Bildern, die in tausende Klassen kategorisiert sind. Die Verwendung von vortrainierten Modellen, die mit ImageNet-Gewichtungen initialisiert sind, bietet den Vorteil, dass das Modell bereits über ein umfangreiches Wissen verfügt und auf ähnliche Aufgaben wie die Fine-Tuning-Aufgabe übertragen werden kann.

##### Kompilieren
Das Kompilieren des Modells ist ein wichtiger Schritt im Trainingsprozess eines neuronalen Netzes, da es dem Mo-dell sagt, wie es während des Trainings aktualisiert werden soll.
Die Parameter, die im model.compile() -Aufruf angegeben sind, haben unterschiedliche Auswirkungen auf das Training des Modells:
optimizer: Der Optimierer bestimmt, wie die Gewichtungen des neuronalen Netzes während des Trainings ange-passt werden sollen, um den Fehler zu minimieren. In diesem Fall wird der Adam-Optimierer verwendet, der ein adaptiver Lernalgorithmus ist, der sich auf jeder Ebene der neuronalen Netze anpasst. Der Adam-Optimierer ver-wendet sowohl die ersten als auch die zweiten Momente der Gradienten, um die Schrittweite der Gewichtungsak-tualisierung zu berechnen.
loss: Die Verlustfunktion bestimmt, wie der Fehler des Modells während des Trainings berechnet wird. In diesem Fall wird der binäre Kreuzentropieverlust verwendet, der speziell für binäre Klassifikationsprobleme geeignet ist. Der binäre Kreuzentropieverlust berechnet den logarithmischen Verlust zwischen den tatsächlichen und vorherge-sagten Klassenwahrscheinlichkeiten.
metrics: Die Metriken geben an, welche Bewertungskriterien verwendet werden sollen, um die Leistung des Mo-dells während des Trainings zu messen. In diesem Fall wird die Genauigkeit (accuracy) als Metrik verwendet. Die Genauigkeit misst den Anteil der korrekt klassifizierten Beispiele.

##### Training : 
Das Modell wird auf einem Trainings-Datensatz (train_dataset) trainiert, mit einer bestimmten Anzahl von Epo-chen (num_epochs). Zusätzlich wird ein Validierungs-Datensatz (validation_dataset) verwendet, um die Leistung des Modells auf unabhängigen Daten während des Trainings zu bewerten.
Das Modell wird mithilfe der Methode "fit" trainiert. Während des Trainings werden die Loss- und Metrik Werte (wie z.B. Genauigkeit) für sowohl den Trainings- als auch den Validierungs-Datensatz aufgezeichnet.
Zusätzlich wird ein Callback verwendet, die während des Trainings aufgerufen werden:
EarlyStopping: Diese Callback-Funktion überwacht die Verlustfunktion (hier mit dem Argument monitor='val_loss') auf dem Validierungs-Datensatz und stoppt das Training, wenn der Validierungsverlust über mehrere Epochen hinweg nicht mehr verbessert wird (hier mit dem Argument patience=3). Dadurch wird das Überanpassen des Mo-dells an die Trainingsdaten vermieden und die allgemeine Leistungsfähigkeit des Modells verbessert.
Das Modell ist nun trainiert und kann verwendet werden, um Drohnen in Echtzeit zu erkennen.

### Parkplatzerkennung 
Die Parkplatzerkennung wurde mit ebenfalls mit einem Maschine-Learning Modell gelöst. 
