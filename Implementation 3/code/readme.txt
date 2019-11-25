1. running algorithm with code: 
python dt_clean.py
python rf_clean.py
python ada_clean.py

2. python dt_clean.py:
(part I)
a. could calculate train and validation accuracy with each depth DT.
b. could return the best accuracy and the depth of the DT.
c. in usual, it takes a few minutes for running out

3. python rf_clean.py:
(part II)
a. could calculate accuracy of training and validation data with d=9, m=10, n=[1,2,5,10,25] of random forest automatically.
b. with just change a number of "m" in rt() function, it could provide different result with different "m", and the rest parameters still are d=9 and n=[1,2,5,10,25].
c. when m is increasing, the running time is also increasing.

4. python ada_clean.py:
(part III)
a. could calculate accuracy of training and validation data with AdaBoost algorithm with L=[1,5,10,20].
b. if open the notation in dl(D) function of "data=data[:2000]" and "y=y[:2000]", the program will return the result of training on the first result, it would increase the running time a lot.
c. so far, i run program for several hours, it still cannot return the result of L=20