This program solves the following problem: students of a school submit preferences for elective classes. However, classes have maximum capacities, thus not everyone can get their first choice of courses.

How can students be assigned to courses in such a way as to maximally adhere to student preferences, while respecting course constraints?

`sorting hat` reads student and course info from files, translates them into a constrained optimization problem which is solved by `ortools`, and finally reports the optimization solution back in the original domain of students and courses.