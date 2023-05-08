# ScheduleOptimizationProblem
This repository contains solution of schedule optimization problem prepared for operational research classes. 

There are 2 input files which could be created by collecting the data through the Google Forms. 
 - The first one stores information about gym users classes types preferences,
 - The second one stores informatio about instructors skills.

First we have implemented Simulated Annealing (https://en.wikipedia.org/wiki/Simulated_annealing) algorithm to optimize the classes organized on the gym such as the profit gained by the gym is maximum.

Then we have created the application which let's choosing the optimization algorithm parameters and input files. We implemented the back-end and front-end of this application using Kivy library.

This is the snipped of app's main menu:
![image](https://user-images.githubusercontent.com/61949638/236826345-d3ccbdad-637d-4920-ad73-c37367223b64.png)

You can choose the algorithm parameters:
![image](https://user-images.githubusercontent.com/61949638/236830612-97c68189-4153-4d11-8f14-faba07f0bee5.png)

Check the optimization progress:
![image](https://user-images.githubusercontent.com/61949638/236830476-edf011d4-ab7d-4c19-bfaf-31f7cf690c92.png)

And see the algorithm overview:
![image](https://user-images.githubusercontent.com/61949638/236830780-8b4f2687-e57c-46e9-9bb9-e195b8595165.png)

And see the result schedule:
![image](https://user-images.githubusercontent.com/61949638/236828770-f34250b1-dae8-4860-ba66-6012fd0e20f8.png)

If you want to get the details of the class you can just click on a time slot of your interest and check the instructor id, qualifications and the participants:
![image](https://user-images.githubusercontent.com/61949638/236829028-ab7753da-7ffe-43b7-948f-6b84adaacc10.png)

If you want to learn more about our app feel free to download the code, in requirements.txt you can find all the needed libraries to run the code.

There are a few thing which could be improved like handling the exception when there are too few timeslots for a given number of participants (e.g. force a user to choose larger number of classrooms).
