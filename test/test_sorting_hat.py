from pathlib import Path
from typing import List

import pytest
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import BoundedLinearExpression, IntVar
from pandas import DataFrame

from pandas.testing import assert_frame_equal

import sorting_hat
from sorting_hat import CourseAssignmentVariables, Courses, StudentPreferences


@pytest.fixture
def course_info():
    return DataFrame([{"name": "course_1", "min_size": 0, "max_size": 1}])


# unsure how to run test, breaks due to not finding example files at path given
# def test_has_example_problem():
#     student_course_preferences: StudentPreferences
#     course_max_nr_students: CourseCapacity
#     student_course_preferences, course_max_nr_students = sorting_hat.get_example_problem()
#     assert len(student_course_preferences) > 0
#     assert len(course_max_nr_students) > 0


def test_creates_assignment_variables():
    students: StudentPreferences = {
        "alice": ["course_1"],
    }
    courses: Courses = Courses(DataFrame([
        {'name': 'course_1', 'min_size': 0, 'max_size': 1},
        {'name': 'course_2', 'min_size': 0, 'max_size': 1},
    ]))
    model = cp_model.CpModel()
    course_assignments: CourseAssignmentVariables = sorting_hat.generate_course_assignment_variables(
        students, courses, model
    )
    expected_variables = CourseAssignmentVariables(
        [
            ("alice", "course_1", model.NewIntVar(0, 1, "alice in course_1")),
            ("alice", "course_2", model.NewIntVar(0, 1, "alice in course_2")),
        ]
    )
    assert course_assignments == expected_variables


def test_gets_cp_variables_by_course_name():
    model: cp_model.CpModel = cp_model.CpModel()
    assignment_data = [
        ("alice", "course_1", model.NewIntVar(0, 1, "alice in course_1")),
        ("alice", "course_2", model.NewIntVar(0, 1, "alice in course_2")),
    ]
    variables: CourseAssignmentVariables = CourseAssignmentVariables(assignment_data)
    var_list_expected: List[IntVar] = [assignment_data[0][2]]
    var_list_returned: List[IntVar] = variables.get_by_course_name("course_1")
    assert var_list_returned == var_list_expected


def test_makes_cost_expression():
    students: StudentPreferences = {
        "alice": ["course_1", "course_2"],
    }
    courses: Courses = Courses(DataFrame([
        {'name': 'course_1', 'min_size': 0, 'max_size': 1},
        {'name': 'course_2', 'min_size': 0, 'max_size': 1},
        {'name': 'course_3', 'min_size': 0, 'max_size': 1},
    ]))
    model = cp_model.CpModel()
    course_assignments: CourseAssignmentVariables = sorting_hat.generate_course_assignment_variables(
        students, courses, model
    )
    alice_in_c1, alice_in_c2 = course_assignments.get_by_student_name_and_courses(
        "alice", ["course_1", "course_2"]
    )
    # note that 'course_3' does not even appear here. we make sure alice can never be assigned to a course she has not listed
    # as a preference through other constraints
    expected_cost: BoundedLinearExpression = 0 * alice_in_c1 + 1 * alice_in_c2
    cost: BoundedLinearExpression = sorting_hat.generate_cost(
        students, course_assignments
    )
    assert cost == expected_cost


def test_gest_all_assignments():
    model: cp_model.CpModel = cp_model.CpModel()
    assignment_data = [
        ("alice", "course_1", model.NewIntVar(0, 1, "alice in course_1")),
        ("alice", "course_2", model.NewIntVar(0, 1, "alice in course_2")),
    ]
    assignments: CourseAssignmentVariables = CourseAssignmentVariables(assignment_data)
    all_assignments_expected: List[IntVar] = [data[2] for data in assignment_data]
    all_assignments: List[IntVar] = assignments.get_all()
    assert all_assignments == all_assignments_expected


def test_reads_student_preference_file():
    pref_file_path: Path = Path("student_preferences.csv")
    preferences: StudentPreferences = sorting_hat.read_student_preferences_file(
        pref_file_path, None
    )
    expected: StudentPreferences = {
        "student_1": ["course_1", "course_2", "course_3"],
        "student_2": ["Difficult, Course, With, Commas, Name"],
    }
    assert preferences == expected


def test_reads_course_capacity_file():
    capacity_file_path: Path = Path("course_capacity.csv")
    capacities: Courses = sorting_hat.Courses.make_from_file(capacity_file_path, None)
    expected: Courses = Courses(
        DataFrame(
            [
                {"name": "course_1", "min_size": 0, "max_size": 1},
                {"name": "course_2", "min_size": 0, "max_size": 1},
                {"name": "course_3", "min_size": 0, "max_size": 1},
                {
                    "name": "Difficult, Course, With, Commas, Name",
                    "min_size": 0,
                    "max_size": 10,
                },
            ]
        )
    )
    assert capacities == expected


def test_solves_problem():
    students: StudentPreferences = {
        "alice": ["course_1",],
        "bob": ["course_2",],
    }
    courses: Courses = Courses(DataFrame([
        {'name': 'course_1', 'min_size': 0, 'max_size': 1},
        {'name': 'course_2', 'min_size': 0, 'max_size': 1},
    ]))
    expected_solution: DataFrame = DataFrame(
        data=[["alice", "course_1"], ["bob", "course_2"]], columns=["student", "course"]
    )
    solution: DataFrame = sorting_hat.solve(students, courses)
    assert_frame_equal(solution, expected_solution)


def test_solves_from_file():
    capacity_path: Path = Path("course_capacity.csv")
    student_path: Path = Path("student_preferences.csv")
    solution_path: Path = Path("test_solution.csv")
    solution_path.unlink(missing_ok=True)
    sorting_hat.solve_from_and_to_files(
        capacity_path, student_path, solution_path, encoding=None
    )
    assert solution_path.is_file()


def test_makes_courses_from_dataframe(course_info):
    courses: Courses = Courses(course_info)
    assert len(courses) == 1


def test_returns_course_names(course_info):
    courses: Courses = Courses(course_info)
    assert courses.get_all_course_names() == ["course_1"]


@pytest.fixture()
def courses(course_info: DataFrame) -> Courses:
    return Courses(course_info)


def test_gets_max_students_by_course_name(courses):
    assert courses.get_max_students_by_course_name('course_1') == 1


def test_raises_exception_when_making_courses_with_incorrect_field_names():
    with pytest.raises(ValueError) as e_info:
        _ = Courses(DataFrame([{'non_existent_field': 1}]))
