Streszczenie 

Wykonano implementacje 3 algolytmów algebry liniowej: mnożenie macierzy, rozkład LU i redukcji wstecznej przy pomocy narzędzi Intel oneAPI [1,2,3]. W celu porównania wydajności wykonano również implementację tych algorytmów w języku C++.
Następnie przeprowadzono analizę wydajności algorytmów z wykorzystaniem narzędzia Intel Advisor na CPU i GPU.
Na podstawie wyników porównano efektywność realizacji obliczeń w wariancie sekwencyjnym i równoległym, a także oceniono wpływ architektury heterogenicznej na uzyskiwaną wydajność.

Słowa kluczowe:	oneApi, obliczenia równoległe, mnożenie macierzy, rozkład LU, redukcja wsteczna

Abstract

The implementation of three linear algebra algorithms was carried out: matrix multiplication, LU decomposition, and back substitution using Intel oneAPI tools [1,2,3]. For performance comparison purposes, implementations of these algorithms were also developed in the C++ programming language. Subsequently, a performance analysis of the algorithms was conducted using the Intel Advisor tool on both CPU and GPU platforms. 
Based on the obtained results, the efficiency of computations in sequential and parallel variants was compared, and the impact of heterogeneous architecture on the achieved performance was evaluated.

Keywords:	oneApi, parallel computation, matrix multiplication, LU decomposition, back substitution
