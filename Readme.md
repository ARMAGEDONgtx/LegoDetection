# PL - Cel
Celem projektu jest policzenie sumarycznej liczby otworów w każdym z widocznych na obrazie obiektów o różnych kształtach i znanych z góry kolorach. Zakładamy, że kąt między osią optyczną kamery a płaszczyzną znacznika (kąt wzniesienia kamery) jest nie większy od 20 stopni (zdjęcia robione z góry). Kolor tła będzie zawsze taki sam, jak na zdjęciu poniżej.

Założenia obowiązujące przy zliczaniu:
każdy obiekt składa się z minimum 2 i maksimum 5 klocków o różnych lub takich samych kolorach,
konfiguracja każdego obiektu jest unikalna w obrębie pojedynczej sceny (tj. obiekt, składający się z dwóch klocków - niebieskiego i białego może pojawić się na danej scenie tylko raz).
liczba obiektów na obrazie jest dowolna,
dostarczony jest plik JSON opisujący z ilu klocków i jakich kolorów składa się każdy z obiektów widocznych na obrazie.


# ENG - Aim 
The goal of the project is to calculate the total number of holes in each of the objects of different shapes and colors known in advance. We assume that the angle between the camera's optical axis and the marker plane (camera elevation angle) is no more than 20 degrees (photos taken from above). The background color will always be the same as the picture below.

Assumptions for counting:
each object consists of a minimum of 2 and a maximum of 5 blocks of different or the same colors,
the configuration of each object is unique within a single scene (i.e. an object consisting of two blocks - blue and white can appear on a given stage only once).
the number of objects in the image is any,
a JSON file is provided describing how many blocks and what colors each of the objects visible in the image consists of.


