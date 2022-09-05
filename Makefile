.PHONY: tests example

tests:
	cd test; PYTHONPATH=../src pytest

example:
	cd src; python sorting_hat.py example_course_capacity.csv example_student_preferences.csv example_assignment_solution.csv
