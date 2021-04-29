Benötigte Bibliotheken
- Tensorflow-gpu 2.3.0
pip install opencv-python
pip install glob2
pip install matplotlib
pip install numpy
pip install pandas
pip install pathlib
pip install DateTime
pip install scikit-learn
pip install scikit-optimize
pip install Pillow
pip install scikit-image
pip install keras
pip install seaborn
pip install joblib

Anleitung zum Trainieren der Netze:
1. Bilder_Labeln\preprocess_images.py: path_cropped, path_augmented, labels_path und
   aug_labels_path müssen neu definiert werden. Nach dem Ausführen dieses Skriptes
   sollten sich ein solcher Aufbau ergeben:
	- Augmented180
	- Cropped
	- Mein_Code (der Ordner bei dem "alles andere" enthalten ist)
		- AlexNet
		- EfficientNet
		...
		- data_pipeline.py
		...

2. Die json-Datei mit den Labels zu den augmentierten Bildern muss im Augmented180-
   Ordner eingefügt werden. Die Datei mit den Labels der vorgeschnittenen Bilder muss
   im Cropped-Ordner eingefügt werden.

3. In data_pipeline.py muss die Variable path umgeschrieben werden. Sie muss auf den
   Ordner zeigen auf dem sich die Augmented180, Cropped und Mein_Code Ordner befinden.
   Nach der Ausführung dieses Skriptes sollten drei TFRecord Datein im Mein_Code-Ordner
   erstellt worden sein: train, val, test. Bei mir sind diese Dateien 2.442.074, 522.727
   und 523.625 KB groß.

4. data_pipeline_names.py erzeugt eine "img_names.csv", welche allerdings bereits in der
   CD enthalten sein sollte.

5. Um die Netze zu trainieren ist kein weiterer Schritt notwendig. Die Skripte können einzeln
   ausgeführt werden. Innerhalb des logs Ordners wird ein Ordner mit dem Namen des Netzes und
   der Uhrzeit erstellt. Dort werden die Trainingsergebnisse und das trainierte Netz gespeichert.
   Im logs Ordner sind aktuell auch noch die Dateien zu meinenen Trainings- und Testrgebnisse.

6. Mithilfe der predict_diff.ipynb Datei können die Netze einzeln getestet und ausgewertet werden.
   Dort müssen zwei Pfade eingestellt werden. Der erste entspricht dem Ordner in dem das Modell
   geispeichert ist, wie z.B.:
   C:\Users\my_user\bwSyncAndShare\Bachelorarbeit-master\Bachelorarbeit-master\logs\VGG16 2021_03_16 T 19-24-05\VGG16
   
   Der zweite Pfad entspricht dem Ordner in dem die restlichen Dateien zu diesem Netz liegen. Es ist
   der gleiche Pfad, aber um eine Ebene weniger. Beim oberen Beispiel würde der zweite Pfad wie
   folgt lauten:
   C:\Users\my_user\bwSyncAndShare\Bachelorarbeit-master\Bachelorarbeit-master\logs\VGG16 2021_03_16 T 19-24-05
   
   Alle durch predict_diff.ipynb erzeugte Dateien werden hier gespeichert. Es werden allerdings nicht
   alle Graphen gespeichert. Die meisten werden nur in Jupyter Notebook angezeigt. Sie können jedoch
   mit plt.savefig gespeichert werden.

7. faulty_imgs.py verwendet den Testdatensatz, um Fehler bei einer festgelegten Toleranz zu
   erkennen. Der Pfad zur Datei an der ein Fehler erkannt wird und die Abweichung zum Soll-Wert
   werden in eine Textdatei ausgegeben. Die Pfade werden analog zu 6. gesetzt.