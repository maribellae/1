ConsoleApplication6 :

1) Загрузить в рабочую директорию файлы first.txt , second.txt - в них лежат матрицы , их можно задать самому в формате (кол-во строк кол-во столбцов и сами данные)

2) В приложении MODE : 0) без параллелизма умножение матриц из файлов. В папке создадутся читаемые виды матриц (first_nice.txt и тд)
                       1) параллелзм нативный
                       2) параллелизм с scheduled(dynamic)-более выгодное распределение ресурсов в цикле
                       3) п. 0) , но задаются автоматически квадратные матрицы выбранного размера
                       4) п. 1) , но задаются автоматически квадратные матрицы выбранного размера
                       5) п. 2) , но задаются автоматически квадратные матрицы выбранного размера
                       6) Алгоритм Штрассена
