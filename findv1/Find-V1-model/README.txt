------------------------------------------
| Requriments                            |
------------------------------------------

This program is written in Python 2.7. It should be compatible with Python later than 2.5 at least. Besides basic Python, Numpy (numpy.scipy.org) and Pylab (matplotlib.sourceforge.net) are required to run this code. 

------------------------------------------
| Running the program                    |
------------------------------------------
Using a terminal or your favourite development tool run the script 'findV1.py'. If you are on Mac/Linux just run the command 'python findV1.py', given that you are in the right directory.

The rest is fairly straight forward, the main menu give you two options. Either you can just freely plot tuning curves under 'Test Model'. That allows you to investigate how different parameters effect the response of our model cell.

The second option, 'Fit the Model', is a bit more challenging but also much more fun. Here you can create a biologically plausible cell using the 'Simulate data' option. The mission is then to figure out the parameters with the lead of the tuning curves. Don't worry, you can get a few parameters to help you on the way. It is actially advisible to get at least one of the time constants for the temporal filter.  Can you find the right parameters is there maybe another set that does a good job of approximating the tuning curves.

Playing around with this gives a deeper understanding to why it can be easy for deterministic fitting algorithms to find a good solution to the fitting task. 
