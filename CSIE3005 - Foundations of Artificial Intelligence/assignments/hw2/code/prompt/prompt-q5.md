## Question 5

Write a better evaluation function for pacman in the provided function betterEvaluationFunction. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did. With depth 2 search, your evaluation function should clear the smallClassic layout with one random ghost more than half the time and still run at a reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he’s winning).

Grading: the autograder will run your agent on the smallClassic layout 10 times. We will assign points to your evaluation function in the following way:

If you win at least once without timing out the autograder, you receive 3 points. Any agent not satisfying these criteria will receive 0 points.
+3 for winning at least 5 times, +6 for winning all 10 times
+2 for an average score of at least 500, +4 for an average score of at least 1000 (including scores on lost games)
+1 if your games take on average less than 30 seconds on the autograder machine, when run with --no-graphics.
The additional points for average score and computation time will only be awarded if you win at least 5 times.
Please do not copy any files from Project 1, as it will not pass.
You can try your agent out under these conditions with

```
python autograder.py -q q5
```

To run it without graphics, use:

```
python autograder.py -q q5 --no-graphics
```
