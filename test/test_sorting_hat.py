from typing import Dict, List

import sorting_hat
from sorting_hat import StudentPreferences, CourseMaxStudents, CourseAssignmentVariables
from ortools.sat.python import cp_model


def test_has_example_problem():
    student_course_preferences: StudentPreferences
    course_max_nr_students: CourseMaxStudents
    student_course_preferences, course_max_nr_students = sorting_hat.get_example_problem()
    assert len(student_course_preferences) > 0
    assert len(course_max_nr_students) > 0


def test_creates_assignment_variables():
    students: StudentPreferences = {
        'alice': ['course_1'],
    }
    courses: CourseMaxStudents = {
        'course_1': 1,
        'course_2': 1,
    }
    model = cp_model.CpModel()
    course_assignments: CourseAssignmentVariables = sorting_hat.generate_course_assignment_variables(
        students,
        courses,
        model
    )
    expected_variables = CourseAssignmentVariables(
        [
            ('alice', 'course_1', model.NewIntVar(0, 1, 'alice in course_1')),
            ('alice', 'course_2', model.NewIntVar(0, 1, 'alice in course_2')),
        ]
    )
    assert course_assignments == expected_variables


# def test_makes_